﻿using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace EncodeServer
{
    public class ClientManager : IUserClient
    {
        private class Client
        {
            private ClientManager manager;
            private TcpClient client;
            private NetworkStream stream;

            public Client(TcpClient client, ClientManager manager)
            {
                this.manager = manager;
                this.client = client;
                this.stream = client.GetStream();
            }

            public async Task Start()
            {
                byte[] idbytes = new byte[2];
                int readBytes = 0;
                try
                {
                    while (true)
                    {
                        readBytes += await stream.ReadAsync(
                            idbytes, readBytes, 2 - readBytes);
                        if (readBytes == 2)
                        {
                            readBytes = 0;
                            var methodId = (RPCMethodId)((idbytes[0] << 8) | idbytes[1]);
                            var argType = RPCTypes.ArgumentTypes[methodId];
                            object arg = null;
                            if (argType != null)
                            {
                                var s = new DataContractSerializer(argType);
                                await Task.Run(() => { arg = s.ReadObject(stream); });
                            }
                            manager.OnRequestReceived(this, (RPCMethodId)methodId, arg);
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Close();
                }
                manager.OnClientClosed(this);
            }

            public void Close()
            {
                if (client != null)
                {
                    client.Close();
                    client = null;
                }
            }

            public NetworkStream GetStream()
            {
                return stream;
            }
        }

        private List<Client> clientList = new List<Client>();
        private List<Task> receiveTask = new List<Task>();

        public async Task Listen()
        {
            int port = int.Parse(ConfigurationManager.AppSettings["Port"]);
            TcpListener listener = new TcpListener(IPAddress.Any, port);
            listener.Start();
            while (true)
            {
                var client = new Client(await listener.AcceptTcpClientAsync(), this);
                receiveTask.Add(client.Start());
                clientList.Add(client);
            }
        }

        private async Task Send(RPCMethodId id, Type type, object obj)
        {
            var idbytes = new byte[2] { (byte)((int)id >> 8), (byte)id };
            var ms = new MemoryStream();
            ms.Write(idbytes, 0, idbytes.Length);
            if(obj != null) {
                var serializer = new DataContractSerializer(type);
                serializer.WriteObject(ms, obj);
            }
            var bytes = ms.ToArray();
            foreach (var client in clientList.ToArray())
            {
                await client.GetStream().WriteAsync(bytes, 0, bytes.Length);
            }
        }

        private IEncodeServer server;

        public ClientManager(IEncodeServer server)
        {
            this.server = server;
        }

        private void OnRequestReceived(Client client, RPCMethodId methodId, object arg)
        {
            switch (methodId)
            {
                case RPCMethodId.SetSetting:
                    server.SetSetting((Setting)arg);
                    break;
                case RPCMethodId.AddQueue:
                    server.AddQueue((string)arg);
                    break;
                case RPCMethodId.RemoveQueue:
                    server.RemoveQueue((string)arg);
                    break;
                case RPCMethodId.PauseEncode:
                    server.PauseEncode((bool)arg);
                    break;
                case RPCMethodId.RequestQueue:
                    server.RequestQueue();
                    break;
                case RPCMethodId.RequestLog:
                    server.RequestLog();
                    break;
                case RPCMethodId.RequestConsole:
                    server.RequestConsole();
                    break;
                case RPCMethodId.RequestLogFile:
                    server.RequestLogFile((LogItem)arg);
                    break;
                case RPCMethodId.RequestState:
                    server.RequestState();
                    break;
            }
        }

        private void OnClientClosed(Client client)
        {
            int index = clientList.IndexOf(client);
            if (index >= 0)
            {
                receiveTask.RemoveAt(index);
                clientList.RemoveAt(index);
            }
        }

        #region IUserClient
        public Task OnQueueData(QueueData data)
        {
            return Send(RPCMethodId.OnQueueData, typeof(QueueData), data);
        }

        public Task OnQueueUpdate(QueueUpdate update)
        {
            return Send(RPCMethodId.OnQueueUpdate, typeof(QueueUpdate), update);
        }

        public Task OnLogData(LogData data)
        {
            return Send(RPCMethodId.OnLogData, typeof(LogData), data);
        }

        public Task OnLogUpdate(LogItem newLog)
        {
            return Send(RPCMethodId.OnLogUpdate, typeof(LogItem), newLog);
        }

        public Task OnConsole(string str)
        {
            return Send(RPCMethodId.OnConsole, typeof(string), str);
        }

        public Task OnConsoleUpdate(string str)
        {
            return Send(RPCMethodId.OnConsoleUpdate, typeof(string), str);
        }

        public Task OnLogFile(string str)
        {
            return Send(RPCMethodId.OnLogFile, typeof(string), str);
        }

        public Task OnState(State state)
        {
            return Send(RPCMethodId.OnState, typeof(State), state);
        }

        public Task OnOperationResult(string result)
        {
            return Send(RPCMethodId.OnOperationResult, typeof(string), result);
        }
        #endregion
    }

    public class EncodeServer : IEncodeServer
    {
        private class TargetDirectory
        {
            public string DirPath { get; private set; }
            public List<string> TsFiles {get;private set;}

            public TargetDirectory(string dirPath)
            {
                DirPath = dirPath;
                TsFiles = Directory.GetFiles(DirPath, "*.ts").ToList();
            }
        }

        [DataContract]
        private class AppData : IExtensibleDataObject
        {
            [DataMember]
            public Setting setting;

            public ExtensionDataObject ExtensionData { get; set; }
        }

        private static string LOG_FILE = "log.xml";
        private static string LOG_DIR = "logs";

        private ClientManager clientManager;
        public Task ServerTask { get; private set; }
        private AppData appData;

        private List<TargetDirectory> queue = new List<TargetDirectory>();
        private LogData log;
        private Queue<string> consoleStrings = new Queue<string>();

        private bool encodePaused = false;
        private bool nowEncoding = false;

        public EncodeServer()
        {
            LoadAppData();
            clientManager = new ClientManager(this);
            ServerTask = clientManager.Listen();
            ReadLog();
        }

        #region Path
        private string GetAmatsukzeCLIPath()
        {
            return Path.Combine(Path.GetDirectoryName(
                typeof(EncodeServer).Assembly.Location), "AmatsukazeCLI.exe");
        }

        private string GetSettingFilePath()
        {
            return "AmatsukazeServer.xml";
        }

        private string GetLogFilePath(long id)
        {
            return LOG_DIR + "\\" + id.ToString("D8") + ".txt";
        }

        private string ReadLogFIle(long id)
        {
            return File.ReadAllText(GetLogFilePath(id));
        }

        private void WriteLogFile(long id, string logstr)
        {
            File.WriteAllText(GetLogFilePath(id), logstr);
        }
        #endregion

        private void LoadAppData()
        {
            if(File.Exists(GetSettingFilePath()) == false)
            {
                appData = new AppData() {
                    setting = new Setting()
                };
                return;
            }
            using (FileStream fs = new FileStream(GetSettingFilePath(), FileMode.Open))
            {
                var s = new DataContractSerializer(typeof(AppData));
                appData = (AppData)s.ReadObject(fs);
                if (appData.setting == null)
                {
                    appData.setting = new Setting();
                }
            }
        }

        private void SaveAppData()
        {
            using (FileStream fs = new FileStream(GetSettingFilePath(), FileMode.Create))
            {
                var s = new DataContractSerializer(typeof(AppData));
                s.WriteObject(fs, appData);
            }
        }

        private void ReadLog()
        {
            try
            {
                if (File.Exists(LOG_FILE))
                {
                    using (FileStream fs = new FileStream(LOG_FILE, FileMode.Open))
                    {
                        var s = new DataContractSerializer(typeof(LogData));
                        log = (LogData)s.ReadObject(fs);
                        if (log.Items == null)
                        {
                            log.Items = new List<LogItem>();
                        }
                    }
                }
            }
            catch (IOException e)
            {
                Console.WriteLine("ログファイルの読み込みに失敗: " + e.Message);
            }
        }

        private void WriteLog()
        {
            try
            {
                using (FileStream fs = new FileStream(LOG_FILE, FileMode.Create))
                {
                    var s = new DataContractSerializer(typeof(LogData));
                    s.WriteObject(fs, log);
                }
            }
            catch (IOException e)
            {
                Console.WriteLine("ログファイル書き込み失敗: " + e.Message);
            }
        }

        private async Task StartEncode()
        {
            nowEncoding = true;
            await Task.Delay(10000);
            nowEncoding = false;
        }

        public Task SetSetting(Setting setting)
        {
            appData.setting = setting;
            SaveAppData();
            return clientManager.OnOperationResult("設定を更新しました");
        }

        public async Task AddQueue(string dirPath)
        {
            if (queue.Find(t => t.DirPath == dirPath) != null)
            {
                await clientManager.OnOperationResult(
                    "すでに同じパスが追加されています。パス:" + dirPath);
                return;
            }
            var target = new TargetDirectory(dirPath);
            if (target.TsFiles.Count == 0)
            {
                await clientManager.OnOperationResult(
                    "エンコード対象ファイルが見つかりません。パス:" + dirPath);
                return;
            }
            queue.Add(target);
            if (nowEncoding == false)
            {
                await StartEncode();
            }
        }

        public async Task RemoveQueue(string dirPath)
        {
            var target = queue.Find(t => t.DirPath == dirPath);
            if (target == null)
            {
                await clientManager.OnOperationResult(
                    "指定されたキューディレクトリが見つかりません。パス:" + dirPath);
                return;
            }
            queue.Remove(target);
        }

        public async Task PauseEncode(bool pause)
        {
            encodePaused = pause;
            if (encodePaused == false && nowEncoding == false)
            {
                await StartEncode();
            }
        }

        public Task RequestQueue()
        {
            QueueData data = new QueueData()
            {
                Items = queue.Select(item => new QueueItem()
                {
                    Path = item.DirPath,
                    MediaFiles = item.TsFiles
                }).ToList()
            };
            return clientManager.OnQueueData(data);
        }

        public Task RequestLog()
        {
            return clientManager.OnLogData(log);
        }

        public Task RequestConsole()
        {
            return clientManager.OnConsole(String.Join("\r\n", consoleStrings));
        }

        public Task RequestLogFile(LogItem item)
        {
            return clientManager.OnLogFile(ReadLogFIle(item.Id));
        }

        public Task RequestState()
        {
            var state = new State() {
                pause = encodePaused
            };
            return clientManager.OnState(state);
        }
    }
}