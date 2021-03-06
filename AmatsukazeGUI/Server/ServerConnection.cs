﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.Serialization;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Amatsukaze.Server
{
    public class ServerConnection : IEncodeServer
    {
        private TcpClient client;
        private NetworkStream stream;
        private IUserClient userClient;
        private Func<string, Task> askServerAddress;
        private string serverIp;
        private int port;
        private bool finished = false;
        private bool reconnect = false;

        public ServerConnection(IUserClient userClient, Func<string, Task> askServerAddress)
        {
            this.userClient = userClient;
            this.askServerAddress = askServerAddress;
        }

        public void SetServerAddress(string serverIp, int port)
        {
            this.serverIp = serverIp;
            this.port = port;
        }

        public void Finish()
        {
            finished = true;
            Close();
        }

        public void Reconnect()
        {
            reconnect = true;
            Close();
        }

        private Task Connect()
        {
            Close();
            client = new TcpClient(serverIp, port);
            Util.AddLog("サーバ(" + serverIp + ":" + port + ")に接続しました");
            stream = client.GetStream();

            // 接続後一通りデータを要求する
            return this.RefreshRequest();
        }

        private void Close()
        {
            if(client != null)
            {
                stream.Close();
                client.Close();
                client = null;
            }
        }

        public async Task Start()
        {
            string failReason = "";
            int failCount = 0;
            int nextWaitSec = 0;
            while (true)
            {
                try
                {
                    if (nextWaitSec > 0)
                    {
                        await Task.Delay(nextWaitSec * 1000);
                        nextWaitSec = 0;
                    }
                    if(serverIp == null)
                    {
                        // 未初期化
                        await askServerAddress("アドレスを入力してください");
                        if(finished)
                        {
                            break;
                        }
                        await Connect();
                    }
                    if(client == null)
                    {
                        // 再接続
                        if (reconnect == false)
                        {
                            await askServerAddress(failReason);
                        }
                        if (finished)
                        {
                            break;
                        }
                        reconnect = false;
                        await Connect();
                    }
                    var rpc = await RPCTypes.Deserialize(stream);
                    OnRequestReceived(rpc.id, rpc.arg);
                    failCount = 0;
                }
                catch (Exception e)
                {
                    // 失敗したら一旦閉じる
                    Close();
                    if (finished)
                    {
                        break;
                    }
                    if (reconnect == false)
                    {
                        nextWaitSec = failCount * 10;
                        Util.AddLog("接続エラー: " + e.Message);
                        Util.AddLog(nextWaitSec.ToString() + "秒後にリトライします");
                        failReason = e.Message;
                        ++failCount;
                    }
                }
            }
        }

        private async Task Send(RPCMethodId id, object obj)
        {
            if(client != null)
            {
                byte[] bytes = RPCTypes.Serialize(id, obj);
                await client.GetStream().WriteAsync(bytes, 0, bytes.Length);
            }
        }

        private void OnRequestReceived(RPCMethodId methodId, object arg)
        {
            switch (methodId)
            {
                case RPCMethodId.OnSetting:
                    userClient.OnSetting((Setting)arg);
                    break;
                case RPCMethodId.OnQueueData:
                    userClient.OnQueueData((QueueData)arg);
                    break;
                case RPCMethodId.OnQueueUpdate:
                    userClient.OnQueueUpdate((QueueUpdate)arg);
                    break;
                case RPCMethodId.OnLogData:
                    userClient.OnLogData((LogData)arg);
                    break;
                case RPCMethodId.OnLogUpdate:
                    userClient.OnLogUpdate((LogItem)arg);
                    break;
                case RPCMethodId.OnConsole:
                    userClient.OnConsole((ConsoleData)arg);
                    break;
                case RPCMethodId.OnConsoleUpdate:
                    userClient.OnConsoleUpdate((ConsoleUpdate)arg);
                    break;
                case RPCMethodId.OnLogFile:
                    userClient.OnLogFile((string)arg);
                    break;
                case RPCMethodId.OnState:
                    userClient.OnState((State)arg);
                    break;
                case RPCMethodId.OnFreeSpace:
                    userClient.OnFreeSpace((DiskFreeSpace)arg);
                    break;
                case RPCMethodId.OnOperationResult:
                    userClient.OnOperationResult((string)arg);
                    break;
            }
        }

        public Task AddQueue(AddQueueDirectory dir)
        {
            return Send(RPCMethodId.AddQueue, dir);
        }

        public Task PauseEncode(bool pause)
        {
            return Send(RPCMethodId.PauseEncode, pause);
        }

        public Task RemoveQueue(string dirPath)
        {
            return Send(RPCMethodId.RemoveQueue, dirPath);
        }

        public Task RequestSetting()
        {
            return Send(RPCMethodId.RequestSetting, null);
        }

        public Task RequestConsole()
        {
            return Send(RPCMethodId.RequestConsole, null);
        }

        public Task RequestLog()
        {
            return Send(RPCMethodId.RequestLog, null);
        }

        public Task RequestLogFile(LogItem item)
        {
            return Send(RPCMethodId.RequestLogFile, item);
        }

        public Task RequestQueue()
        {
            return Send(RPCMethodId.RequestQueue, null);
        }

        public Task RequestState()
        {
            return Send(RPCMethodId.RequestState, null);
        }

        public Task RequestFreeSpace()
        {
            return Send(RPCMethodId.RequestFreeSpace, null);
        }

        public Task SetSetting(Setting setting)
        {
            return Send(RPCMethodId.SetSetting, setting);
        }
    }
}
