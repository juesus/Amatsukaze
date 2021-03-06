/**
* MPEG2-TS splitter
* Copyright (c) 2017 Nekopanda
*
* This software is released under the MIT License.
* http://opensource.org/licenses/mit-license.php
*/
#pragma once

#include "common.h"

#include <algorithm>
#include <vector>
#include <map>
#include <array>

#include "StreamUtils.hpp"
#include "Mpeg2TsParser.hpp"
#include "Mpeg2VideoParser.hpp"
#include "H264VideoParser.hpp"
#include "AdtsParser.hpp"
#include "Mpeg2PsWriter.hpp"
#include "WaveWriter.h"

class VideoFrameParser : public AMTObject, public PesParser {
public:
	VideoFrameParser(AMTContext&ctx)
		: AMTObject(ctx)
		, PesParser()
		, videoStreamFormat(VS_MPEG2)
		, videoFormat()
		, mpeg2parser()
		, h264parser(ctx)
		, parser(&mpeg2parser)
	{ }

	void setStreamFormat(VIDEO_STREAM_FORMAT streamFormat) {
		if (videoStreamFormat != streamFormat) {
			switch (streamFormat) {
			case VS_MPEG2:
				parser = &mpeg2parser;
				break;
			case VS_H264:
				parser = &h264parser;
				break;
			}
			reset();
			videoStreamFormat = streamFormat;
		}
	}

	VIDEO_STREAM_FORMAT getStreamFormat() { return videoStreamFormat; }

	void reset() {
		videoFormat = VideoFormat();
		parser->reset();
	}

protected:

	virtual void onPesPacket(int64_t clock, PESPacket packet) {
		if (clock == -1) {
			ctx.error("Video PES Packet にクロック情報がありません");
			return;
		}
		if (!packet.has_PTS()) {
			ctx.error("Video PES Packet に PTS がありません");
			return;
		}

		int64_t PTS = packet.has_PTS() ? packet.PTS : -1;
		int64_t DTS = packet.has_DTS() ? packet.DTS : PTS;
		MemoryChunk payload = packet.paylod();

		if (!parser->inputFrame(payload, frameInfo, PTS, DTS)) {
			ctx.error("フレーム情報の取得に失敗 PTS=%lld", PTS);
			return;
		}

		if (frameInfo.size() > 0) {
			const VideoFrameInfo& frame = frameInfo[0];

			if (frame.format.isEmpty()) {
				// フォーマットがわからないとデコードできないので流さない
				return;
			}

			if (frame.format != videoFormat) {
				// フォーマットが変わった
				videoFormat = frame.format;
				onVideoFormatChanged(frame.format);
			}

			onVideoPesPacket(clock, frameInfo, packet);
		}
	}

	virtual void onVideoPesPacket(int64_t clock, const std::vector<VideoFrameInfo>& frames, PESPacket packet) = 0;

	virtual void onVideoFormatChanged(VideoFormat fmt) = 0;

private:
	VIDEO_STREAM_FORMAT videoStreamFormat;
	VideoFormat videoFormat;
	
	std::vector<VideoFrameInfo> frameInfo;

	MPEG2VideoParser mpeg2parser;
	H264VideoParser h264parser;

	IVideoParser* parser;

};

class AudioFrameParser : public AMTObject, public PesParser {
public:
	AudioFrameParser(AMTContext&ctx)
		: AMTObject(ctx)
		, PesParser()
		, adtsParser(ctx)
	{ }

	virtual void onPesPacket(int64_t clock, PESPacket packet) {
		if (clock == -1) {
			ctx.error("Audio PES Packet にクロック情報がありません");
			return;
		}

		int64_t PTS = packet.has_PTS() ? packet.PTS : -1;
		int64_t DTS = packet.has_DTS() ? packet.DTS : PTS;
		MemoryChunk payload = packet.paylod();

		adtsParser.inputFrame(payload, frameData, PTS);

		if (frameData.size() > 0) {
			const AudioFrameData& frame = frameData[0];

			if (frame.format != format) {
				// フォーマットが変わった
				format = frame.format;
				onAudioFormatChanged(frame.format);
			}

			onAudioPesPacket(clock, frameData, packet);
		}
	}

	virtual void onAudioPesPacket(int64_t clock, const std::vector<AudioFrameData>& frames, PESPacket packet) = 0;

	virtual void onAudioFormatChanged(AudioFormat fmt) = 0;

private:
	AudioFormat format;

	std::vector<AudioFrameData> frameData;

	AdtsParser adtsParser;
};

// TSストリームを一定量だけ戻れるようにする
class TsPacketBuffer : public TsPacketParser {
public:
	TsPacketBuffer(AMTContext& ctx)
		: TsPacketParser(ctx)
		, handler(NULL)
		, numBefferedPackets_(0)
		, numMaxPackets(0)
		, buffering(false)
	{ }

	void setHandler(TsPacketHandler* handler) {
		this->handler = handler;
	}

	int numBefferedPackets() {
		return numBefferedPackets_;
	}

	void clearBuffer() {
		buffer.clear();
		numBefferedPackets_ = 0;
	}

	void setEnableBuffering(bool enable) {
		buffering = enable;
		if (!buffering) {
			clearBuffer();
		}
	}

	void setNumBufferingPackets(int numPackets) {
		numMaxPackets = numPackets;
	}

	void backAndInput() {
		if (handler != NULL) {
			for (int i = 0; i < buffer.size(); i += TS_PACKET_LENGTH) {
				TsPacket packet(buffer.get() + i);
				if (packet.parse() && packet.check()) {
					handler->onTsPacket(-1, packet);
				}
			}
		}
	}

	virtual void onTsPacket(TsPacket packet) {
		if (buffering) {
			if (numBefferedPackets_ >= numMaxPackets) {
				buffer.trimHead((numMaxPackets - numBefferedPackets_ + 1) * TS_PACKET_LENGTH);
				numBefferedPackets_ = numMaxPackets - 1;
			}
			buffer.add(packet.data, TS_PACKET_LENGTH);
			++numBefferedPackets_;
		}
		if (handler != NULL) {
			handler->onTsPacket(-1, packet);
		}
	}

private:
	TsPacketHandler* handler;
	AutoBuffer buffer;
	int numBefferedPackets_;
	int numMaxPackets;
	bool buffering;
};

class TsSystemClock {
public:
	TsSystemClock()
		: PcrPid(-1)
		, numPcrReceived(0)
		, numTotakPacketsReveived(0)
		, pcrInfo()
	{ }

	void setPcrPid(int PcrPid) {
		this->PcrPid = PcrPid;
	}

	// 十分な数のPCRを受信したか
	bool pcrReceived() {
		return numPcrReceived >= 2;
	}

	// 現在入力されたパケットを基準にしてrelativeだけ後のパケットの入力時刻を返す
	int64_t getClock(int relative) {
		if (!pcrReceived()) {
			return -1;
		}
		int index = numTotakPacketsReveived + relative - 1;
		int64_t clockDiff = pcrInfo[1].clock - pcrInfo[0].clock;
		int64_t indexDiff = pcrInfo[1].packetIndex - pcrInfo[0].packetIndex;
		return clockDiff * (index - pcrInfo[1].packetIndex) / indexDiff + pcrInfo[1].clock;
	}

	// TSストリームを最初から読み直すときに呼び出す
	void backTs() {
		numTotakPacketsReveived = 0;
	}

	// TSストリームの全データを入れること
	void inputTsPacket(TsPacket packet) {
		if (packet.PID() == PcrPid) {
			if (packet.has_adaptation_field()) {
				MemoryChunk data = packet.adapdation_field();
				AdapdationField af(data.data, (int)data.length);
				if (af.parse() && af.check()) {
					if (af.discontinuity_indicator()) {
						// PCRが連続でないのでリセット
						numPcrReceived = 0;
					}
					if (pcrInfo[1].packetIndex < numTotakPacketsReveived) {
						std::swap(pcrInfo[0], pcrInfo[1]);
						if (af.PCR_flag()) {
							pcrInfo[1].clock = af.program_clock_reference;
							pcrInfo[1].packetIndex = numTotakPacketsReveived;
							++numPcrReceived;
						}

						// テスト用
						//if (pcrReceived()) {
						//	PRINTF("PCR: %f Mbps\n", currentBitrate() / (1024 * 1024));
						//}
					}
				}
			}
		}
		++numTotakPacketsReveived;
	}

	double currentBitrate() {
		int clockDiff = int(pcrInfo[1].clock - pcrInfo[0].clock);
		int indexDiff = int(pcrInfo[1].packetIndex - pcrInfo[0].packetIndex);
		return (double)(indexDiff * TS_PACKET_LENGTH * 8) / clockDiff * 27000000;
	}

private:
	struct PCR_Info {
		int64_t clock = 0;
		int packetIndex = -1;
	};

	int PcrPid;
	int numPcrReceived;
	int numTotakPacketsReveived;
	PCR_Info pcrInfo[2];
};

class TsSplitter : public AMTObject, protected TsPacketSelectorHandler {
public:
	TsSplitter(AMTContext& ctx)
		: AMTObject(ctx)
		, initPhase(PMT_WAITING)
		, tsPacketHandler(*this)
		, pcrDetectionHandler(*this)
		, tsPacketParser(ctx)
		, tsPacketSelector(ctx)
		, videoParser(ctx, *this)
	{
		tsPacketParser.setHandler(&tsPacketHandler);
		tsPacketParser.setNumBufferingPackets(50 * 1024); // 9.6MB
		tsPacketSelector.setHandler(this);
		reset();
	}

	void reset() {
		initPhase = PMT_WAITING;
		preferedServiceId = -1;
		selectedServiceId = -1;
		tsPacketParser.setEnableBuffering(true);
	}

	// 0以下で指定無効
	void setServiceId(int sid) {
		preferedServiceId = sid;
	}

	int getActualServiceId(int sid) {
		return selectedServiceId;
	}

	void inputTsData(MemoryChunk data) {
		tsPacketParser.inputTS(data);
	}
	void flush() {
		tsPacketParser.flush();
	}

protected:
	enum INITIALIZATION_PHASE {
		PMT_WAITING,	// PAT,PMT待ち
		PCR_WAITING,	// ビットレート取得のためPCR2つを受信中
		INIT_FINISHED,	// 必要な情報は揃った
	};

	class SpTsPacketHandler : public TsPacketHandler {
		TsSplitter& this_;
	public:
		SpTsPacketHandler(TsSplitter& this_)
			: this_(this_) { }

		virtual void onTsPacket(int64_t clock, TsPacket packet) {
			this_.tsSystemClock.inputTsPacket(packet);

			int64_t packetClock = this_.tsSystemClock.getClock(0);
			this_.tsPacketSelector.inputTsPacket(packetClock, packet);
		}
	};
	class PcrDetectionHandler : public TsPacketHandler {
		TsSplitter& this_;
	public:
		PcrDetectionHandler(TsSplitter& this_)
			: this_(this_) { }

		virtual void onTsPacket(int64_t clock, TsPacket packet) {
			this_.tsSystemClock.inputTsPacket(packet);
			if (this_.tsSystemClock.pcrReceived()) {
				this_.ctx.debug("必要な情報は取得したのでTSを最初から読み直します");
				this_.initPhase = INIT_FINISHED;
				// ハンドラを戻して最初から読み直す
				this_.tsPacketParser.setHandler(&this_.tsPacketHandler);
				this_.tsPacketSelector.resetParser();
				this_.tsSystemClock.backTs();

				int64_t startClock = this_.tsSystemClock.getClock(0);
				this_.ctx.info("開始Clock: %lld", startClock);
				this_.tsPacketSelector.setStartClock(startClock);
				
				this_.tsPacketParser.backAndInput();
				// もう必要ないのでバッファリングはOFF
				this_.tsPacketParser.setEnableBuffering(false);
			}
		}
	};
	class SpVideoFrameParser : public VideoFrameParser {
		TsSplitter& this_;
	public:
		SpVideoFrameParser(AMTContext&ctx, TsSplitter& this_)
			: VideoFrameParser(ctx), this_(this_) { }

	protected:
		virtual void onVideoPesPacket(int64_t clock, const std::vector<VideoFrameInfo>& frames, PESPacket packet) {
			this_.onVideoPesPacket(clock, frames, packet);
		}

		virtual void onVideoFormatChanged(VideoFormat fmt) {
			this_.onVideoFormatChanged(fmt);
		}
	};
	class SpAudioFrameParser : public AudioFrameParser {
		TsSplitter& this_;
		int audioIdx;
	public:
		SpAudioFrameParser(AMTContext&ctx, TsSplitter& this_, int audioIdx)
			: AudioFrameParser(ctx), this_(this_), audioIdx(audioIdx) { }

	protected:
		virtual void onAudioPesPacket(int64_t clock, const std::vector<AudioFrameData>& frames, PESPacket packet) {
			this_.onAudioPesPacket(audioIdx, clock, frames, packet);
		}

		virtual void onAudioFormatChanged(AudioFormat fmt) {
			this_.onAudioFormatChanged(audioIdx, fmt);
		}
	};

	INITIALIZATION_PHASE initPhase;

	TsPacketBuffer tsPacketParser;
	TsSystemClock tsSystemClock;
	SpTsPacketHandler tsPacketHandler;
	PcrDetectionHandler pcrDetectionHandler;
	TsPacketSelector tsPacketSelector;

	SpVideoFrameParser videoParser;
	std::vector<SpAudioFrameParser*> audioParsers;

	int preferedServiceId;
	int selectedServiceId;

	virtual void onVideoPesPacket(
		int64_t clock,
		const std::vector<VideoFrameInfo>& frames,
		PESPacket packet) = 0;

	virtual void onVideoFormatChanged(VideoFormat fmt) = 0;

	virtual void onAudioPesPacket(
		int audioIdx,
		int64_t clock,
		const std::vector<AudioFrameData>& frames,
		PESPacket packet) = 0;

	virtual void onAudioFormatChanged(int audioIdx, AudioFormat fmt) = 0;

	// サービスを設定する場合はサービスのpids上でのインデックス
	// なにもしない場合は負の値の返す
	virtual int onPidSelect(int TSID, const std::vector<int>& pids) {
		ctx.info("[PAT更新]");
		for (int i = 0; i < int(pids.size()); ++i) {
			if (preferedServiceId == pids[i]) {
				selectedServiceId = pids[i];
				ctx.info("サービス 0x%04x を選択", selectedServiceId);
				return i;
			}
		}
		if (preferedServiceId > 0) {
			// サービス指定があるのに該当サービスがなかったらエラーとする
			std::ostringstream ss;
			ss << "サービスID: ";
			for (int i = 0; i < (int)pids.size(); ++i) {
				if (i > 0) ss << ", ";
				ss << pids[i];
			}
			ss << " 指定サービスID: " << preferedServiceId;
			ctx.error("指定されたサービスがありません");
			ctx.error(ss.str().c_str());
			THROW(InvalidOperationException, "failed to select service");
		}
		selectedServiceId = pids[0];
		ctx.info("サービス 0x%04x を選択（指定がありませんでした）", selectedServiceId);
		return 0;
	}

	virtual void onPmtUpdated(int PcrPid) {
		if (initPhase == PMT_WAITING) {
			initPhase = PCR_WAITING;
			// PCRハンドラに置き換えてTSを最初から読み直す
			tsPacketParser.setHandler(&pcrDetectionHandler);
			tsSystemClock.setPcrPid(PcrPid);
			tsPacketSelector.resetParser();
			tsSystemClock.backTs();
			tsPacketParser.backAndInput();
		}
	}

	// TsPacketSelectorでPID Tableが変更された時変更後の情報が送られる
	virtual void onPidTableChanged(const PMTESInfo video, const std::vector<PMTESInfo>& audio) {
		// 映像ストリーム形式をセット
		switch (video.stype) {
		case 0x02: // MPEG2-VIDEO
			videoParser.setStreamFormat(VS_MPEG2);
			break;
		case 0x1B: // H.264/AVC
			videoParser.setStreamFormat(VS_H264);
			break;
		}

		// 必要な数だけ音声パーサ作る
		size_t numAudios = audio.size();
		while (audioParsers.size() < numAudios) {
			int audioIdx = int(audioParsers.size());
			audioParsers.push_back(new SpAudioFrameParser(ctx, *this, audioIdx));
			ctx.info("音声パーサ %d を追加", audioIdx);
		}
	}

	virtual void onVideoPacket(int64_t clock, TsPacket packet) {
		videoParser.onTsPacket(clock, packet);
	}

	virtual void onAudioPacket(int64_t clock, TsPacket packet, int audioIdx) {
		ASSERT(audioIdx < audioParsers.size());
		audioParsers[audioIdx]->onTsPacket(clock, packet);
	}
};

