/**
* Transcode manager
* Copyright (c) 2017 Nekopanda
*
* This software is released under the MIT License.
* http://opensource.org/licenses/mit-license.php
*/
#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <memory>
#include <limits>
#include <direct.h>

#include "StreamUtils.hpp"
#include "TsSplitter.hpp"
#include "Transcode.hpp"
#include "VideoFilter.hpp"
#include "StreamReform.hpp"
#include "PacketCache.hpp"

// �J���[�X�y�[�X��`���g������
#include "libavutil/pixfmt.h"

namespace av {

// �J���[�X�y�[�X3�Z�b�g
// x265�͐��l���̂܂܂ł�OK�����Ax264��help���������string�łȂ����
// �Ȃ�Ȃ��悤�Ȃ̂ŕϊ����`
// �Ƃ肠����ARIB STD-B32 v3.7�ɏ����Ă���̂���

// 3���F
static const char* getColorPrimStr(int color_prim) {
	switch (color_prim) {
	case AVCOL_PRI_BT709: return "bt709";
	case AVCOL_PRI_BT2020: return "bt2020";
	default:
		THROWF(FormatException,
			"Unsupported color primaries (%d)", color_prim);
	}
	return NULL;
}

// �K���}
static const char* getTransferCharacteristicsStr(int transfer_characteritics) {
	switch (transfer_characteritics) {
	case AVCOL_TRC_BT709: return "bt709";
	case AVCOL_TRC_IEC61966_2_4: return "iec61966-2-4";
	case AVCOL_TRC_BT2020_10: return "bt2020-10";
	case AVCOL_TRC_SMPTEST2084: return "smpte-st-2084";
	case AVCOL_TRC_ARIB_STD_B67: return "arib-std-b67";
	default:
		THROWF(FormatException,
			"Unsupported color transfer characteritics (%d)", transfer_characteritics);
	}
	return NULL;
}

// �ϊ��W��
static const char* getColorSpaceStr(int color_space) {
	switch (color_space) {
	case AVCOL_SPC_BT709: return "bt709";
	case AVCOL_SPC_BT2020_NCL: return "bt2020nc";
	default:
		THROWF(FormatException,
			"Unsupported color color space (%d)", color_space);
	}
	return NULL;
}

} // namespace av {

enum ENUM_ENCODER {
	ENCODER_X264,
	ENCODER_X265,
	ENCODER_QSVENC,
	ENCODER_NVENC,
	ENCODER_FFMPEG,
};

struct BitrateSetting {
  double a, b;
  double h264;
  double h265;

  double getTargetBitrate(VIDEO_STREAM_FORMAT format, double srcBitrate) const {
    double base = a * srcBitrate + b;
    if (format == VS_H264) {
      return base * h264;
    }
    else if (format == VS_H265) {
      return base * h265;
    }
    return base;
  }
};

static const char* encoderToString(ENUM_ENCODER encoder) {
	switch (encoder) {
	case ENCODER_X264: return "x264";
	case ENCODER_X265: return "x265";
	case ENCODER_QSVENC: return "QSVEnc";
	case ENCODER_NVENC: return "NVEnc";
	case ENCODER_FFMPEG: return "FFmpeg";
	}
	return "Unknown";
}

static std::string makeEncoderArgs(
	ENUM_ENCODER encoder,
	const std::string& binpath,
	const std::string& options,
	const VideoFormat& fmt,
	const std::string& outpath)
{
	std::ostringstream ss;

	ss << "\"" << binpath << "\"";

	if (encoder == ENCODER_FFMPEG) {
		ss << " -i - " << options;
		return ss.str();
	}

	// y4m�w�b�_�ɂ���̂ŕK�v�Ȃ�
	//ss << " --fps " << fmt.frameRateNum << "/" << fmt.frameRateDenom;
	//ss << " --input-res " << fmt.width << "x" << fmt.height;
	//ss << " --sar " << fmt.sarWidth << ":" << fmt.sarHeight;

	if (fmt.colorPrimaries != AVCOL_PRI_UNSPECIFIED) {
		ss << " --colorprim " << av::getColorPrimStr(fmt.colorPrimaries);
	}
	if (fmt.transferCharacteristics != AVCOL_TRC_UNSPECIFIED) {
		ss << " --transfer " << av::getTransferCharacteristicsStr(fmt.transferCharacteristics);
	}
	if (fmt.colorSpace != AVCOL_TRC_UNSPECIFIED) {
		ss << " --colormatrix " << av::getColorSpaceStr(fmt.colorSpace);
	}

	// �C���^�[���[�X
	switch (encoder) {
	case ENCODER_X264:
	case ENCODER_QSVENC:
	case ENCODER_NVENC:
		ss << (fmt.progressive ? "" : " --tff");
		break;
	case ENCODER_X265:
		ss << (fmt.progressive ? " --no-interlace" : " --interlace tff");
		break;
	}

	ss << " " << options << " -o \"" << outpath << "\"";

	// ���͌`��
	switch (encoder) {
	case ENCODER_X264:
		ss << " --demuxer y4m -";
		break;
	case ENCODER_X265:
		ss << " --y4m --input -";
		break;
	case ENCODER_QSVENC:
	case ENCODER_NVENC:
		ss << " --format raw --y4m -i -";
		break;
	case ENCODER_FFMPEG:
		ss << " -i -";
		break;
	}

	return ss.str();
}

static std::string makeMuxerArgs(
	const std::string& binpath,
	const std::string& inVideo,
	const VideoFormat& videoFormat,
	const std::vector<std::string>& inAudios,
	const std::string& outpath)
{
	std::ostringstream ss;

	ss << "\"" << binpath << "\"";
	if (videoFormat.fixedFrameRate) {
		ss << " -i \"" << inVideo << "?fps="
			 << videoFormat.frameRateNum << "/"
			 << videoFormat.frameRateDenom << "\"";
	}
	else {
		ss << " -i \"" << inVideo << "\"";
	}
	for (const auto& inAudio : inAudios) {
		ss << " -i \"" << inAudio << "\"";
	}
	ss << " --optimize-pd";
	ss << " -o \"" << outpath << "\"";

	return ss.str();
}

static std::string makeTimelineEditorArgs(
	const std::string& binpath,
	const std::string& inpath,
	const std::string& outpath,
	const std::string& timecodepath,
	std::pair<int, int> timebase)
{
	std::ostringstream ss;
	ss << "\"" << binpath << "\"";
	ss << " --track 1";
	ss << " --timecode \"" << timecodepath << "\"";
	ss << " --media-timescale " << timebase.first;
	ss << " --media-timebase " << timebase.second;
	ss << " \"" << inpath << "\"";
	ss << " \"" << outpath << "\"";
	return ss.str();
}

inline bool ends_with(std::string const & value, std::string const & ending)
{
	if (ending.size() > value.size()) return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

enum AMT_CLI_MODE {
  AMT_CLI_TS,
  AMT_CLI_GENERIC,
};

class TempDirectory : AMTObject, NonCopyable
{
public:
	TempDirectory(AMTContext& ctx, const std::string& tmpdir)
		: AMTObject(ctx)
	{
		for (int code = (int)time(NULL) & 0xFFFFFF; code > 0; ++code) {
			auto path = genPath(tmpdir, code);
			if (_mkdir(path.c_str()) == 0) {
				path_ = path;
				break;
			}
			if (errno != EEXIST) {
				break;
			}
		}
		if (path_.size() == 0) {
			THROW(IOException, "�ꎞ�f�B���N�g���쐬���s");
		}
	}
	~TempDirectory() {
		// �ꎞ�t�@�C�����폜
		ctx.clearTmpFiles();
		// �f�B���N�g���폜
		if (_rmdir(path_.c_str()) != 0) {
			ctx.warn("�ꎞ�f�B���N�g���폜�Ɏ��s: ", path_.c_str());
		}
	}

	std::string path() const {
		return path_;
	}

private:
	std::string path_;

	std::string genPath(const std::string& base, int code)
	{
		std::ostringstream ss;
		ss << base << "/amt" << code;
		return ss.str();
	}
};

class TranscoderSetting : public AMTObject
{
public:
	TranscoderSetting(
			AMTContext& ctx,
			std::string workDir,
			AMT_CLI_MODE mode,
			std::string srcFilePath,
			std::string outVideoPath,
			std::string outInfoJsonPath,
			ENUM_ENCODER encoder,
			std::string encoderPath,
			std::string encoderOptions,
			std::string muxerPath,
			std::string timelineditorPath,
			bool twoPass,
			bool autoBitrate,
		  bool pulldown,
			BitrateSetting bitrate,
			int serviceId,
			DECODER_TYPE mpeg2decoder,
			DECODER_TYPE h264decoder,
			bool dumpStreamInfo)
		: AMTObject(ctx)
		, tmpDir(ctx, workDir)
		, mode(mode)
		, srcFilePath(srcFilePath)
		, outVideoPath(outVideoPath)
		, outInfoJsonPath(outInfoJsonPath)
		, encoder(encoder)
		, encoderPath(encoderPath)
		, encoderOptions(encoderOptions)
		, muxerPath(muxerPath)
		, timelineditorPath(timelineditorPath)
		, twoPass(twoPass)
		, autoBitrate(autoBitrate)
		, pulldown(pulldown)
		, bitrate(bitrate)
		, serviceId(serviceId)
		, mpeg2decoder(mpeg2decoder)
		, h264decoder(h264decoder)
		, dumpStreamInfo(dumpStreamInfo)
	{
		//
	}

	AMT_CLI_MODE getMode() const {
		return mode;
	}

	std::string getSrcFilePath() const {
		return srcFilePath;
	}

	std::string getOutInfoJsonPath() const {
		return outInfoJsonPath;
	}

	ENUM_ENCODER getEncoder() const {
		return encoder;
	}

	std::string getEncoderPath() const {
		return encoderPath;
	}

	std::string getMuxerPath() const {
		return muxerPath;
	}

	std::string getTimelineEditorPath() const {
		return timelineditorPath;
	}

	bool isTwoPass() const {
		return twoPass;
	}

	bool isAutoBitrate() const {
		return autoBitrate;
	}

	bool isPulldownEnabled() const {
		return pulldown;
	}

	BitrateSetting getBitrate() const {
		return bitrate;
	}

	int getServiceId() const {
		return serviceId;
	}

	DECODER_TYPE getMpeg2Decoder() const {
		return mpeg2decoder;
	}

	DECODER_TYPE getH264Decoder() const {
		return h264decoder;
	}

	bool isDumpStreamInfo() const {
		return dumpStreamInfo;
	}

	std::string getAudioFilePath() const
	{
		std::ostringstream ss;
		ss << tmpDir.path() << "/audio.dat";
		ctx.registerTmpFile(ss.str());
		return ss.str();
	}

	std::string getIntVideoFilePath(int index) const
	{
		std::ostringstream ss;
		ss << tmpDir.path() << "/i" << index << ".mpg";
		ctx.registerTmpFile(ss.str());
		return ss.str();
	}

	std::string getStreamInfoPath() const
	{
		return outVideoPath + "-streaminfo.dat";
	}

	std::string getEncVideoFilePath(int vindex, int index) const
	{
		std::ostringstream ss;
		ss << tmpDir.path() << "/v" << vindex << "-" << index << ".raw";
		ctx.registerTmpFile(ss.str());
		return ss.str();
  }

  std::string getEncStatsFilePath(int vindex, int index) const
  {
    std::ostringstream ss;
    ss << tmpDir.path() << "/s" << vindex << "-" << index << ".log";
		ctx.registerTmpFile(ss.str());
		// x264��.mbtree����������̂�
		ctx.registerTmpFile(ss.str() + ".mbtree");
    return ss.str();
  }

	std::string getEncTimecodeFilePath(int vindex, int index) const
	{
		std::ostringstream ss;
		ss << tmpDir.path() << "/tc" << vindex << "-" << index << ".txt";
		ctx.registerTmpFile(ss.str());
		return ss.str();
	}

	std::string getEncPulldownFilePath(int vindex, int index) const
	{
		std::ostringstream ss;
		ss << tmpDir.path() << "/pd" << vindex << "-" << index << ".txt";
		ctx.registerTmpFile(ss.str());
		return ss.str();
	}

	std::string getIntAudioFilePath(int vindex, int index, int aindex) const
	{
		std::ostringstream ss;
		ss << tmpDir.path() << "/a" << vindex << "-" << index << "-" << aindex << ".aac";
		ctx.registerTmpFile(ss.str());
		return ss.str();
	}

	std::string getVfrTmpFilePath(int index) const
	{
		std::ostringstream ss;
		ss << tmpDir.path() << "/t" << index << ".mp4";
		ctx.registerTmpFile(ss.str());
		return ss.str();
	}

	std::string getOutFilePath(int index) const
	{
		std::ostringstream ss;
		ss << outVideoPath;
		if (index != 0) {
			ss << "-" << index;
		}
		ss << ".mp4";
		return ss.str();
	}

	std::string getOutSummaryPath() const
	{
		std::ostringstream ss;
		ss << outVideoPath;
		ss << ".txt";
		return ss.str();
	}

  std::string getOptions(
    VIDEO_STREAM_FORMAT srcFormat, double srcBitrate, bool pulldown,
    int pass, int vindex, int index) const 
  {
    std::ostringstream ss;
    ss << encoderOptions;
    if (autoBitrate) {
      double targetBitrate = bitrate.getTargetBitrate(srcFormat, srcBitrate);
			if (encoder == ENCODER_QSVENC) {
				ss << " --la " << (int)targetBitrate;
				ss << " --maxbitrate " << (int)(targetBitrate * 2);
			}
			else if (encoder == ENCODER_NVENC) {
				ss << " --vbrhq " << (int)targetBitrate;
				ss << " --maxbitrate " << (int)(targetBitrate * 2);
			}
			else {
				ss << " --bitrate " << (int)targetBitrate;
				ss << " --vbv-maxrate " << (int)(targetBitrate * 2);
				ss << " --vbv-bufsize 31250"; // high profile level 4.1
			}
    }
		if (pulldown) {
			ss << " --pdfile-in \"" << getEncPulldownFilePath(vindex, index) << "\"";
		}
    if (pass >= 0) {
      ss << " --pass " << pass;
      ss << " --stats \"" << getEncStatsFilePath(vindex, index) << "\"";
    }
    return ss.str();
  }

	void dump() const {
		ctx.info("[�ݒ�]");
    ctx.info("Mode: %s", (mode == AMT_CLI_TS) ? "�ʏ�" : "��ʃt�@�C�����[�h");
    ctx.info("Input: %s", srcFilePath.c_str());
		ctx.info("Output: %s", outVideoPath.c_str());
		ctx.info("WorkDir: %s", tmpDir.path().c_str());
		ctx.info("OutJson: %s", outInfoJsonPath.c_str());
		ctx.info("Encoder: %s", encoderToString(encoder));
		ctx.info("EncoderPath: %s", encoderPath.c_str());
		ctx.info("EncoderOptions: %s", encoderOptions.c_str());
		ctx.info("MuxerPath: %s", muxerPath.c_str());
    ctx.info("TimelineeditorPath: %s", timelineditorPath.c_str());
    ctx.info("autoBitrate: %s", autoBitrate ? "yes" : "no");
    ctx.info("Bitrate: %f:%f:%f", bitrate.a, bitrate.b, bitrate.h264);
    ctx.info("twoPass: %s", twoPass ? "yes" : "no");
		if (serviceId > 0) {
			ctx.info("ServiceId: 0x%04x", serviceId);
		}
		else {
			ctx.info("ServiceId: �w��Ȃ�");
		}
		ctx.info("mpeg2decoder: %s", decoderToString(mpeg2decoder));
		ctx.info("h264decoder: %s", decoderToString(h264decoder));
	}

private:
	TempDirectory tmpDir;

	AMT_CLI_MODE mode;
	// ���̓t�@�C���p�X�i�g���q���܂ށj
	std::string srcFilePath;
	// �o�̓t�@�C���p�X�i�g���q�������j
	std::string outVideoPath;
	// ���ʏ��JSON�o�̓p�X
	std::string outInfoJsonPath;
	// �G���R�[�_�ݒ�
	ENUM_ENCODER encoder;
	std::string encoderPath;
	std::string encoderOptions;
	std::string muxerPath;
	std::string timelineditorPath;
	bool twoPass;
	bool autoBitrate;
	bool pulldown;
	BitrateSetting bitrate;
	int serviceId;
	DECODER_TYPE mpeg2decoder;
	DECODER_TYPE h264decoder;
	// �f�o�b�O�p�ݒ�
	bool dumpStreamInfo;

	const char* decoderToString(DECODER_TYPE decoder) const {
		switch (decoder) {
		case DECODER_QSV: return "QSV";
		case DECODER_CUVID: return "CUVID";
		}
		return "default";
	}
};

class AMTSplitter : public TsSplitter {
public:
	AMTSplitter(AMTContext& ctx, const TranscoderSetting& setting)
		: TsSplitter(ctx)
		, setting_(setting)
		, psWriter(ctx)
		, writeHandler(*this)
		, audioFile_(setting.getAudioFilePath(), "wb")
		, videoFileCount_(0)
		, videoStreamType_(-1)
		, audioStreamType_(-1)
		, audioFileSize_(0)
		, srcFileSize_(0)
	{
		psWriter.setHandler(&writeHandler);
	}

	StreamReformInfo split()
	{
		writeHandler.resetSize();

		readAll();

		// for debug
		printInteraceCount();

		return StreamReformInfo(ctx, videoFileCount_,
			videoFrameList_, audioFrameList_, streamEventList_);
	}

	int64_t getSrcFileSize() const {
		return srcFileSize_;
	}

	int64_t getTotalIntVideoSize() const {
		return writeHandler.getTotalSize();
	}

protected:
	class StreamFileWriteHandler : public PsStreamWriter::EventHandler {
		TsSplitter& this_;
		std::unique_ptr<File> file_;
		int64_t totalIntVideoSize_;
	public:
		StreamFileWriteHandler(TsSplitter& this_)
			: this_(this_), totalIntVideoSize_() { }
		virtual void onStreamData(MemoryChunk mc) {
			if (file_ != NULL) {
				file_->write(mc);
				totalIntVideoSize_ += mc.length;
			}
		}
		void open(const std::string& path) {
			file_ = std::unique_ptr<File>(new File(path, "wb"));
		}
		void close() {
			file_ = nullptr;
		}
		void resetSize() {
			totalIntVideoSize_ = 0;
		}
		int64_t getTotalSize() const {
			return totalIntVideoSize_;
		}
	};

	const TranscoderSetting& setting_;
	PsStreamWriter psWriter;
	StreamFileWriteHandler writeHandler;
	File audioFile_;

	int videoFileCount_;
	int videoStreamType_;
	int audioStreamType_;
	int64_t audioFileSize_;
	int64_t srcFileSize_;

	// �f�[�^
  std::vector<VideoFrameInfo> videoFrameList_;
	std::vector<FileAudioFrameInfo> audioFrameList_;
	std::vector<StreamEvent> streamEventList_;

	void readAll() {
		enum { BUFSIZE = 4 * 1024 * 1024 };
		auto buffer_ptr = std::unique_ptr<uint8_t[]>(new uint8_t[BUFSIZE]);
		MemoryChunk buffer(buffer_ptr.get(), BUFSIZE);
		File srcfile(setting_.getSrcFilePath(), "rb");
		srcFileSize_ = srcfile.size();
		size_t readBytes;
		do {
			readBytes = srcfile.read(buffer);
			inputTsData(MemoryChunk(buffer.data, readBytes));
		} while (readBytes == buffer.length);
	}

	static bool CheckPullDown(PICTURE_TYPE p0, PICTURE_TYPE p1) {
		switch (p0) {
		case PIC_TFF:
		case PIC_BFF_RFF:
			return (p1 == PIC_TFF || p1 == PIC_TFF_RFF);
		case PIC_BFF:
		case PIC_TFF_RFF:
			return (p1 == PIC_BFF || p1 == PIC_BFF_RFF);
		default: // ����ȊO�̓`�F�b�N�ΏۊO
			return true;
		}
	}

	void printInteraceCount() {

		if (videoFrameList_.size() == 0) {
			ctx.error("�t���[��������܂���");
			return;
		}

		// ���b�v�A���E���h���Ȃ�PTS�𐶐�
		std::vector<std::pair<int64_t, int>> modifiedPTS;
		int64_t videoBasePTS = videoFrameList_[0].PTS;
		int64_t prevPTS = videoFrameList_[0].PTS;
		for (int i = 0; i < int(videoFrameList_.size()); ++i) {
			int64_t PTS = videoFrameList_[i].PTS;
			int64_t modPTS = prevPTS + int64_t((int32_t(PTS) - int32_t(prevPTS)));
			modifiedPTS.emplace_back(modPTS, i);
			prevPTS = modPTS;
		}

		// PTS�Ń\�[�g
		std::sort(modifiedPTS.begin(), modifiedPTS.end());

#if 0
		// �t���[�����X�g���o��
		FILE* framesfp = fopen("frames.txt", "w");
		fprintf(framesfp, "FrameNumber,DecodeFrameNumber,PTS,Duration,FRAME_TYPE,PIC_TYPE,IsGOPStart\n");
		for (int i = 0; i < (int)modifiedPTS.size(); ++i) {
			int64_t PTS = modifiedPTS[i].first;
			int decodeIndex = modifiedPTS[i].second;
			const VideoFrameInfo& frame = videoFrameList_[decodeIndex];
			int PTSdiff = -1;
			if (i < (int)modifiedPTS.size() - 1) {
				int64_t nextPTS = modifiedPTS[i + 1].first;
				const VideoFrameInfo& nextFrame = videoFrameList_[modifiedPTS[i + 1].second];
				PTSdiff = int(nextPTS - PTS);
				if (CheckPullDown(frame.pic, nextFrame.pic) == false) {
					ctx.warn("Flag Check Error: PTS=%lld %s -> %s",
						PTS, PictureTypeString(frame.pic), PictureTypeString(nextFrame.pic));
				}
			}
			fprintf(framesfp, "%d,%d,%lld,%d,%s,%s,%d\n",
				i, decodeIndex, PTS, PTSdiff, FrameTypeString(frame.type), PictureTypeString(frame.pic), frame.isGopStart ? 1 : 0);
		}
		fclose(framesfp);
#endif

		// PTS�Ԋu���o��
		struct Integer {
			int v;
			Integer() : v(0) { }
		};

		std::array<int, MAX_PIC_TYPE> interaceCounter = { 0 };
		std::map<int, Integer> PTSdiffMap;
		prevPTS = -1;
		for (const auto& ptsIndex : modifiedPTS) {
			int64_t PTS = ptsIndex.first;
			const VideoFrameInfo& frame = videoFrameList_[ptsIndex.second];
			interaceCounter[(int)frame.pic]++;
			if (prevPTS != -1) {
				int PTSdiff = int(PTS - prevPTS);
				PTSdiffMap[PTSdiff].v++;
			}
			prevPTS = PTS;
		}

		ctx.info("[�f���t���[�����v���]");

		int64_t totalTime = modifiedPTS.back().first - videoBasePTS;
		ctx.info("����: %f �b", totalTime / 90000.0);

		ctx.info("FRAME=%d DBL=%d TLP=%d TFF=%d BFF=%d TFF_RFF=%d BFF_RFF=%d",
			interaceCounter[0], interaceCounter[1], interaceCounter[2], interaceCounter[3], interaceCounter[4], interaceCounter[5], interaceCounter[6]);

		for (const auto& pair : PTSdiffMap) {
			ctx.info("(PTS_Diff,Cnt)=(%d,%d)", pair.first, pair.second.v);
		}
	}

	// TsSplitter���z�֐� //

	virtual void onVideoPesPacket(
		int64_t clock,
		const std::vector<VideoFrameInfo>& frames,
		PESPacket packet)
	{
		for (const VideoFrameInfo& frame : frames) {
      videoFrameList_.push_back(frame);
		}
		psWriter.outVideoPesPacket(clock, frames, packet);
	}

	virtual void onVideoFormatChanged(VideoFormat fmt) {
		ctx.info("[�f���t�H�[�}�b�g�ύX]");
		if (fmt.fixedFrameRate) {
			ctx.info("�T�C�Y: %dx%d FPS: %d/%d", fmt.width, fmt.height, fmt.frameRateNum, fmt.frameRateDenom);
		}
		else {
			ctx.info("�T�C�Y: %dx%d FPS: VFR", fmt.width, fmt.height);
		}

		// �o�̓t�@�C����ύX
		writeHandler.open(setting_.getIntVideoFilePath(videoFileCount_++));
		psWriter.outHeader(videoStreamType_, audioStreamType_);

		StreamEvent ev = StreamEvent();
		ev.type = VIDEO_FORMAT_CHANGED;
		ev.frameIdx = (int)videoFrameList_.size();
		streamEventList_.push_back(ev);
	}

	virtual void onAudioPesPacket(
		int audioIdx, 
		int64_t clock, 
		const std::vector<AudioFrameData>& frames, 
		PESPacket packet)
	{
		for (const AudioFrameData& frame : frames) {
			FileAudioFrameInfo info = frame;
			info.audioIdx = audioIdx;
			info.codedDataSize = frame.codedDataSize;
			info.fileOffset = audioFileSize_;
			audioFile_.write(MemoryChunk(frame.codedData, frame.codedDataSize));
			audioFileSize_ += frame.codedDataSize;
			audioFrameList_.push_back(info);
		}
		if (videoFileCount_ > 0) {
			psWriter.outAudioPesPacket(audioIdx, clock, frames, packet);
		}
	}

	virtual void onAudioFormatChanged(int audioIdx, AudioFormat fmt) {
		ctx.info("[����%d�t�H�[�}�b�g�ύX]", audioIdx);
		ctx.info("�`�����l��: %s �T���v�����[�g: %d",
			getAudioChannelString(fmt.channels), fmt.sampleRate);

		StreamEvent ev = StreamEvent();
		ev.type = AUDIO_FORMAT_CHANGED;
		ev.audioIdx = audioIdx;
		ev.frameIdx = (int)audioFrameList_.size();
		streamEventList_.push_back(ev);
	}

	// TsPacketSelectorHandler���z�֐� //

	virtual void onPidTableChanged(const PMTESInfo video, const std::vector<PMTESInfo>& audio) {
		// �x�[�X�N���X�̏���
		TsSplitter::onPidTableChanged(video, audio);

		ASSERT(audio.size() > 0);
		videoStreamType_ = video.stype;
		audioStreamType_ = audio[0].stype;

		StreamEvent ev = StreamEvent();
		ev.type = PID_TABLE_CHANGED;
		ev.numAudio = (int)audio.size();
		ev.frameIdx = (int)videoFrameList_.size();
		streamEventList_.push_back(ev);
	}
};

class RFFExtractor
{
public:
	void clear() {
		prevFrame_ = nullptr;
	}

	void inputFrame(av::EncodeWriter& encoder, std::unique_ptr<av::Frame>&& frame, PICTURE_TYPE pic) {

		// PTS��inputFrame�ōĒ�`�����̂ŏC�����Ȃ��ł��̂܂ܓn��
		switch (pic) {
		case PIC_FRAME:
		case PIC_TFF:
		case PIC_TFF_RFF:
			encoder.inputFrame(*frame);
			break;
		case PIC_FRAME_DOUBLING:
			encoder.inputFrame(*frame);
			encoder.inputFrame(*frame);
			break;
		case PIC_FRAME_TRIPLING:
			encoder.inputFrame(*frame);
			encoder.inputFrame(*frame);
			encoder.inputFrame(*frame);
			break;
		case PIC_BFF:
			encoder.inputFrame(*mixFields(
				(prevFrame_ != nullptr) ? *prevFrame_ : *frame, *frame));
			break;
		case PIC_BFF_RFF:
			encoder.inputFrame(*mixFields(
				(prevFrame_ != nullptr) ? *prevFrame_ : *frame, *frame));
			encoder.inputFrame(*frame);
			break;
		}

		prevFrame_ = std::move(frame);
	}

private:
	std::unique_ptr<av::Frame> prevFrame_;

	// 2�̃t���[���̃g�b�v�t�B�[���h�A�{�g���t�B�[���h������
	static std::unique_ptr<av::Frame> mixFields(av::Frame& topframe, av::Frame& bottomframe)
	{
		auto dstframe = std::unique_ptr<av::Frame>(new av::Frame(-1));

		AVFrame* top = topframe();
		AVFrame* bottom = bottomframe();
		AVFrame* dst = (*dstframe)();

		// �t���[���̃v���p�e�B���R�s�[
		av_frame_copy_props(dst, top);

		// �������T�C�Y�Ɋւ�������R�s�[
		dst->format = top->format;
		dst->width = top->width;
		dst->height = top->height;

		// �������m��
		if (av_frame_get_buffer(dst, 64) != 0) {
			THROW(RuntimeException, "failed to allocate frame buffer");
		}

		const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get((AVPixelFormat)(dst->format));
		int pixel_shift = (desc->comp[0].depth > 8) ? 1 : 0;
		int nplanes = (dst->format != AV_PIX_FMT_NV12) ? 3 : 2;

		for (int i = 0; i < nplanes; ++i) {
			int hshift = (i > 0 && dst->format != AV_PIX_FMT_NV12) ? desc->log2_chroma_w : 0;
			int vshift = (i > 0) ? desc->log2_chroma_h : 0;
			int wbytes = (dst->width >> hshift) << pixel_shift;
			int height = dst->height >> vshift;

			for (int y = 0; y < height; y += 2) {
				uint8_t* dst0 = dst->data[i] + dst->linesize[i] * (y + 0);
				uint8_t* dst1 = dst->data[i] + dst->linesize[i] * (y + 1);
				uint8_t* src0 = top->data[i] + top->linesize[i] * (y + 0);
				uint8_t* src1 = bottom->data[i] + bottom->linesize[i] * (y + 1);
				memcpy(dst0, src0, wbytes);
				memcpy(dst1, src1, wbytes);
			}
		}

		return std::move(dstframe);
	}
};

static PICTURE_TYPE getPictureTypeFromAVFrame(AVFrame* frame)
{
	bool interlaced = frame->interlaced_frame != 0;
	bool tff = frame->top_field_first != 0;
	int repeat = frame->repeat_pict;
	if (interlaced == false) {
		switch (repeat) {
		case 0: return PIC_FRAME;
		case 1: return tff ? PIC_TFF_RFF : PIC_BFF_RFF;
		case 2: return PIC_FRAME_DOUBLING;
		case 4: return PIC_FRAME_TRIPLING;
		default: THROWF(FormatException, "Unknown repeat count: %d", repeat);
		}
		return PIC_FRAME;
	}
	else {
		if (repeat) {
			THROW(FormatException, "interlaced and repeat ???");
		}
		return tff ? PIC_TFF : PIC_BFF;
	}
}

struct EncodeFileInfo {
  double srcBitrate;
  double targetBitrate;
};

class AMTVideoEncoder : public AMTObject {
public:
	AMTVideoEncoder(
		AMTContext&ctx,
		const TranscoderSetting& setting,
		StreamReformInfo& reformInfo)
		: AMTObject(ctx)
		, setting_(setting)
		, reformInfo_(reformInfo)
		, prevFrameIndex_()
		, pd_data_(NULL)
		, filterTop_(NULL)
		, filterBottom_(NULL)
		, preFilter_(this, true)
		, postFilter_(this, false)
	{
		addFilter(&threads_[0]);
		addFilter(&preFilter_);
		addFilter(&threads_[1]);
		if (true) {
			addFilter(&cnrFilter_);
			addFilter(&threads_[2]);
		}
		//addFilter(&postFilter_);
	}

	~AMTVideoEncoder() {
		delete[] encoders_; encoders_ = NULL;
	}

  std::vector<EncodeFileInfo> peform(int videoFileIndex)
  {
		videoFileIndex_ = videoFileIndex;
    numEncoders_ = reformInfo_.getNumEncoders(videoFileIndex);
    efi_ = std::vector<EncodeFileInfo>(numEncoders_, EncodeFileInfo());

		const auto& format0 = reformInfo_.getFormat(0, videoFileIndex_);
		int bufsize = format0.videoFormat.width * format0.videoFormat.height * 3;

		// pulldown�t�@�C������
		bool pulldown = (setting_.isPulldownEnabled() && reformInfo_.hasRFF());
		ctx.setCounter("pulldown", (int)pulldown);
		if (pulldown) {
			generatePulldownFile(bufsize);
		}

		// x265�ŃC���^���[�X�̏ꍇ�̓t�B�[���h���[�h
		bool fieldMode = 
			(setting_.getEncoder() == ENCODER_X265 &&
			 format0.videoFormat.progressive == false);
		ctx.setCounter("fieldmode", (int)fieldMode);

    // �r�b�g���[�g�v�Z
    double srcBitrate = getSourceBitrate();
    ctx.info("���͉f���r�b�g���[�g: %d kbps", (int)srcBitrate);

    VIDEO_STREAM_FORMAT srcFormat = reformInfo_.getVideoStreamFormat();
    double targetBitrate = std::numeric_limits<float>::quiet_NaN();
    if (setting_.isAutoBitrate()) {
      targetBitrate = setting_.getBitrate().getTargetBitrate(srcFormat, srcBitrate);
      ctx.info("�ڕW�f���r�b�g���[�g: %d kbps", (int)targetBitrate);
    }
    for (int i = 0; i < numEncoders_; ++i) {
      efi_[i].srcBitrate = srcBitrate;
      efi_[i].targetBitrate = targetBitrate;
    }

		auto getOptions = [&](int pass, int index) {
			return setting_.getOptions(
				srcFormat, srcBitrate, pulldown, pass, videoFileIndex_, index);
		};

    if (setting_.isTwoPass()) {
      ctx.info("1/2�p�X �G���R�[�h�J�n");
      processAllData(fieldMode, bufsize, getOptions, 1);
      ctx.info("2/2�p�X �G���R�[�h�J�n");
      processAllData(fieldMode, bufsize, getOptions, 2);
    }
    else {
      processAllData(fieldMode, bufsize, getOptions, -1);
    }

    return efi_;
	}

private:
	class SpVideoReader : public av::VideoReader {
	public:
		SpVideoReader(AMTVideoEncoder* this_)
			: VideoReader(this_->ctx)
			, this_(this_)
		{ }
  protected:
    virtual void onVideoFormat(AVStream *stream, VideoFormat fmt) {
			this_->nrFilter_.init(5, 7, !fmt.progressive);
			this_->cnrFilter_.init(5, 7, 8, !fmt.progressive);
		}
    virtual void onFrameDecoded(av::Frame& frame) {
			// �t���[�����R�s�[���ăt�B���^�ɓn��
			this_->filterTop_->onFrame(std::unique_ptr<av::Frame>(new av::Frame(frame)));
    }
    virtual void onAudioPacket(AVPacket& packet) { }
	private:
		AMTVideoEncoder* this_;
	};

	class ThreadFilter : public VideoFilter, private DataPumpThread<std::unique_ptr<av::Frame>> {
	public:
		ThreadFilter() : DataPumpThread(8)
		{ }
		virtual void start() {
			DataPumpThread<std::unique_ptr<av::Frame>>::start();
		}
		virtual void onFrame(std::unique_ptr<av::Frame>&& frame) {
			put(std::move(frame), 1);
		}
		virtual void finish() {
			join();
		}
	private:
		virtual void OnDataReceived(std::unique_ptr<av::Frame>&& data) {
			sendFrame(std::move(data));
		}
	};

	class PrePostFilter : public VideoFilter {
	public:
		PrePostFilter(AMTVideoEncoder* this_, bool pre) : this_(this_), pre_(pre) { }
		virtual void onFrame(std::unique_ptr<av::Frame>&& frame) {
			if (pre_) {
				if (this_->preFilterFrame(*frame)) {
					// NV12 -> YV12�ϊ�
					sendFrame(this_->toYV12(std::move(frame)));
				}
			}
			else {
				this_->postFilterFrame(std::move(frame));
			}
		}
		virtual void start() { }
		virtual void finish() { }
	private:
		AMTVideoEncoder* this_;
		bool pre_;
	};

	const TranscoderSetting& setting_;
	StreamReformInfo& reformInfo_;

	int videoFileIndex_;
  int numEncoders_;
	av::EncodeWriter* encoders_;
	std::stringstream* pd_data_;
  std::vector<EncodeFileInfo> efi_;
	
	VideoFilter* filterTop_;
	VideoFilter* filterBottom_;

	ThreadFilter threads_[4];
	PrePostFilter preFilter_;
	PrePostFilter postFilter_;
	TemporalNRFilter nrFilter_;
	CudaTemporalNRFilter cnrFilter_;

	int prevFrameIndex_;

	RFFExtractor rffExtractor_;

	void addFilter(VideoFilter* filter)
	{
		if (filterTop_ == NULL) {
			filterTop_ = filterBottom_ = filter;
		}
		else {
			filterBottom_->nextFilter = filter;
			filterBottom_ = filter;
		}
	}
  void processAllData(bool fieldMode, int bufsize, std::function<std::string(int,int)> getOptions, int pass)
  {
    // ������
    encoders_ = new av::EncodeWriter[numEncoders_];
    SpVideoReader reader(this);

    for (int i = 0; i < numEncoders_; ++i) {
      const auto& format = reformInfo_.getFormat(i, videoFileIndex_);
      std::string args = makeEncoderArgs(
        setting_.getEncoder(),
        setting_.getEncoderPath(),
        getOptions(pass, i),
        format.videoFormat,
				setting_.getEncVideoFilePath(videoFileIndex_, i));
      ctx.info("[�G���R�[�_�J�n]");
      ctx.info(args.c_str());
      encoders_[i].start(args, format.videoFormat, fieldMode, bufsize);
    }

    // �t�B���^�J�n
		for (auto f = filterTop_; f != NULL; f = f->nextFilter) {
			f->start();
		}

    // �G���R�[�h
    std::string intVideoFilePath = setting_.getIntVideoFilePath(videoFileIndex_);
    reader.readAll(intVideoFilePath,
			setting_.getMpeg2Decoder(), setting_.getH264Decoder(), DECODER_DEFAULT);

    // �t�B���^���I�����Ď����Ɉ����p��
		for (auto f = filterTop_; f != NULL; f = f->nextFilter) {
			f->finish();
		}

    // �c�����t���[��������
    for (int i = 0; i < numEncoders_; ++i) {
      encoders_[i].finish();
    }

    // �I��
		rffExtractor_.clear();
    delete[] encoders_; encoders_ = NULL;
  }

	void generatePulldownFile(int bufsize)
	{
		// ������
		pd_data_ = new std::stringstream[numEncoders_];
		SpVideoReader reader(this);

		// �t�B���^�J�n
		for (auto filter = filterTop_; filter != NULL; filter = filter->nextFilter) {
			filter->start();
		}

		// �G���R�[�h
		std::string intVideoFilePath = setting_.getIntVideoFilePath(videoFileIndex_);
		reader.readAll(intVideoFilePath,
			setting_.getMpeg2Decoder(), setting_.getH264Decoder(), DECODER_DEFAULT);

		// �t�B���^���I�����Ď����Ɉ����p��
		for (auto filter = filterTop_; filter != NULL; filter = filter->nextFilter) {
			filter->finish();
		}

		// �t�@�C���o��
		for (int i = 0; i < numEncoders_; ++i) {
			std::string str = pd_data_[i].str();
			MemoryChunk mc(reinterpret_cast<uint8_t*>(const_cast<char*>(str.data())), str.size());
			File file(setting_.getEncPulldownFilePath(videoFileIndex_, i), "w");
			file.write(mc);
		}

		// �I��
		delete[] pd_data_; pd_data_ = NULL;
	}

  double getSourceBitrate()
  {
    // �r�b�g���[�g�v�Z
    VIDEO_STREAM_FORMAT srcFormat = reformInfo_.getVideoStreamFormat();
    int64_t srcBytes = 0, srcDuration = 0;
    for (int i = 0; i < numEncoders_; ++i) {
      const auto& info = reformInfo_.getSrcVideoInfo(i, videoFileIndex_);
      srcBytes += info.first;
      srcDuration += info.second;
    }
    return ((double)srcBytes * 8 / 1000) / ((double)srcDuration / MPEG_CLOCK_HZ);
  }

	const char* toPulldownFlag(PICTURE_TYPE pic, bool progressive) {
		switch (pic) {
		case PIC_FRAME: return "SGL";
		case PIC_FRAME_DOUBLING: return "DBL";
		case PIC_FRAME_TRIPLING: return "TPL";
		case PIC_TFF: return progressive ? "PTB" : "TB";
		case PIC_BFF: return progressive ? "PBT" : "BT";
		case PIC_TFF_RFF: return "TBT";
		case PIC_BFF_RFF: return "BTB";
		default: THROWF(FormatException, "Unknown PICTURE_TYPE %d", pic);
		}
		return NULL;
	}

	static std::unique_ptr<av::Frame> toYV12(std::unique_ptr<av::Frame>&& frame)
	{
		if ((*frame)()->format != AV_PIX_FMT_NV12) {
			return std::move(frame);
		}

		//printf("f");

		auto dstframe = std::unique_ptr<av::Frame>(new av::Frame(frame->frameIndex_));

		AVFrame* src = (*frame)();
		AVFrame* dst = (*dstframe)();

		// �t���[���̃v���p�e�B���R�s�[
		av_frame_copy_props(dst, src);

		// �������T�C�Y�Ɋւ�������R�s�[
		dst->format = AV_PIX_FMT_YUV420P;
		dst->width = src->width;
		dst->height = src->height;

		// �������m��
		if (av_frame_get_buffer(dst, 64) != 0) {
			THROW(RuntimeException, "failed to allocate frame buffer");
		}

		// copy luna
		for (int y = 0; y < dst->height; ++y) {
			uint8_t* dst0 = dst->data[0] + dst->linesize[0] * y;
			uint8_t* src0 = src->data[0] + src->linesize[0] * y;
			memcpy(dst0, src0, dst->width);
		}

		// extract chroma
		__m128i shuffle_i = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
		int chroma_h = dst->height / 2;
		for (int y = 0; y < chroma_h; ++y) {
			uint8_t* dstU = dst->data[1] + dst->linesize[1] * y;
			uint8_t* dstV = dst->data[2] + dst->linesize[2] * y;
			uint8_t* src0 = src->data[1] + src->linesize[1] * y;

			int x = 0;
			int chroma_w = dst->width / 2;
			for (; x <= (chroma_w - 8); x += 8) {
				__m128i s = _mm_loadu_si128((__m128i*)&src0[x * 2]);
				__m128i ss = _mm_shuffle_epi8(s, shuffle_i);
				*(int64_t*)&dstU[x] = _mm_extract_epi64(ss, 0);
				*(int64_t*)&dstV[x] = _mm_extract_epi64(ss, 1);
			}
			for (; x < chroma_w; ++x) {
				dstU[x] = src0[x * 2 + 0];
				dstV[x] = src0[x * 2 + 1];
			}
		}

		return std::move(dstframe);
	}

	bool preFilterFrame(av::Frame& frame)
	{
		// ffmpeg���ǂ�pts��wrap���邩������Ȃ��̂œ��̓f�[�^��
		// ����33bit�݂̂�����
		//�i26���Ԉȏ゠�铮�悾�Əd������\���͂��邪�����j
		int64_t pts = frame()->pts & ((int64_t(1) << 33) - 1);

		int frameIndex = reformInfo_.getVideoFrameIndex(pts, videoFileIndex_);
		if (frameIndex == -1) {
			frameIndex = prevFrameIndex_;
			ctx.incrementCounter("incident");
			ctx.warn("Unknown PTS frame %lld", pts);
		}
		else {
			prevFrameIndex_ = frameIndex;
		}

		frame.frameIndex_ = frameIndex;

		if (pd_data_ != NULL) {
			// pulldown�t�@�C��������
			VideoFrameInfo info = reformInfo_.getVideoFrameInfo(frameIndex);
			int encoderIndex = reformInfo_.getEncoderIndex(frameIndex);
			auto& ss = pd_data_[encoderIndex];
			ss << toPulldownFlag(info.pic, info.progressive) << std::endl;
			return false;
		}

		return true;
	}

	void postFilterFrame(std::unique_ptr<av::Frame>&& frame)
	{
		int frameIndex = frame->frameIndex_;

		VideoFrameInfo info = reformInfo_.getVideoFrameInfo(frameIndex);
		int encoderIndex = reformInfo_.getEncoderIndex(frameIndex);

		auto& encoder = encoders_[encoderIndex];

		if (reformInfo_.isVFR() || setting_.isPulldownEnabled()) {
			// VFR�̏ꍇ�͕K���P�������o��
			// �v���_�E�����L���ȏꍇ�̓t���O�ŏ�������̂łP�������o��
			encoder.inputFrame(*frame);
		}
		else {
			// RFF�t���O����
			rffExtractor_.inputFrame(encoder, std::move(frame), info.pic);
		}

		reformInfo_.frameEncoded(frameIndex);
	}

};

class AMTSimpleVideoEncoder : public AMTObject {
public:
  AMTSimpleVideoEncoder(
    AMTContext& ctx,
    const TranscoderSetting& setting)
    : AMTObject(ctx)
    , setting_(setting)
    , reader_(this)
    , thread_(this, 8)
  {
    //
  }

  void encode()
  {
    if (setting_.isTwoPass()) {
      ctx.info("1/2�p�X �G���R�[�h�J�n");
      processAllData(1);
      ctx.info("2/2�p�X �G���R�[�h�J�n");
      processAllData(2);
    }
    else {
      processAllData(-1);
    }
  }

	int getAudioCount() const {
		return audioCount_;
	}

	int64_t getSrcFileSize() const {
		return srcFileSize_;
	}

	VideoFormat getVideoFormat() const {
		return videoFormat_;
	}

private:
  class SpVideoReader : public av::VideoReader {
  public:
    SpVideoReader(AMTSimpleVideoEncoder* this_)
      : VideoReader(this_->ctx)
      , this_(this_)
    { }
  protected:
		virtual void onFileOpen(AVFormatContext *fmt) {
			this_->onFileOpen(fmt);
		}
    virtual void onVideoFormat(AVStream *stream, VideoFormat fmt) {
      this_->onVideoFormat(stream, fmt);
    }
    virtual void onFrameDecoded(av::Frame& frame) {
      this_->onFrameDecoded(frame);
    }
    virtual void onAudioPacket(AVPacket& packet) {
      this_->onAudioPacket(packet);
    }
  private:
    AMTSimpleVideoEncoder* this_;
  };

  class SpDataPumpThread : public DataPumpThread<std::unique_ptr<av::Frame>> {
  public:
    SpDataPumpThread(AMTSimpleVideoEncoder* this_, int bufferingFrames)
      : DataPumpThread(bufferingFrames)
      , this_(this_)
    { }
  protected:
    virtual void OnDataReceived(std::unique_ptr<av::Frame>&& data) {
      this_->onFrameReceived(std::move(data));
    }
  private:
    AMTSimpleVideoEncoder* this_;
  };

	class AudioFileWriter : public av::AudioWriter {
	public:
		AudioFileWriter(AVStream* stream, const std::string& filename, int bufsize)
			: AudioWriter(stream, bufsize)
			, file_(filename, "wb")
		{ }
	protected:
		virtual void onWrite(MemoryChunk mc) {
			file_.write(mc);
		}
	private:
		File file_;
	};

  const TranscoderSetting& setting_;
  SpVideoReader reader_;
  av::EncodeWriter* encoder_;
  SpDataPumpThread thread_;

	int audioCount_;
	std::vector<std::unique_ptr<AudioFileWriter>> audioFiles_;
	std::vector<int> audioMap_;

	int64_t srcFileSize_;
	VideoFormat videoFormat_;
	RFFExtractor rffExtractor_;

  int pass_;

	void onFileOpen(AVFormatContext *fmt)
	{
		audioMap_ = std::vector<int>(fmt->nb_streams, -1);
		if (pass_ <= 1) { // 2�p�X�ڂ͏o�͂��Ȃ�
			audioCount_ = 0;
			for (int i = 0; i < (int)fmt->nb_streams; ++i) {
				if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
					audioFiles_.emplace_back(new AudioFileWriter(
						fmt->streams[i], setting_.getIntAudioFilePath(0, 0, audioCount_), 8 * 1024));
					audioMap_[i] = audioCount_++;
				}
			}
		}
	}

  void processAllData(int pass)
  {
    pass_ = pass;

		encoder_ = new av::EncodeWriter();

    // �G���R�[�h�X���b�h�J�n
    thread_.start();

    // �G���R�[�h
    reader_.readAll(setting_.getSrcFilePath(),
			setting_.getMpeg2Decoder(), setting_.getH264Decoder(), DECODER_DEFAULT);

    // �G���R�[�h�X���b�h���I�����Ď����Ɉ����p��
    thread_.join();

    // �c�����t���[��������
    encoder_->finish();

		if (pass_ <= 1) { // 2�p�X�ڂ͏o�͂��Ȃ�
			for (int i = 0; i < audioCount_; ++i) {
				audioFiles_[i]->flush();
			}
			audioFiles_.clear();
		}

		rffExtractor_.clear();
		audioMap_.clear();
		delete encoder_; encoder_ = NULL;
  }

  void onVideoFormat(AVStream *stream, VideoFormat fmt)
  {
		videoFormat_ = fmt;

    // �r�b�g���[�g�v�Z
    File file(setting_.getSrcFilePath(), "rb");
		srcFileSize_ = file.size();
    double srcBitrate = ((double)srcFileSize_ * 8 / 1000) / (stream->duration * av_q2d(stream->time_base));
    ctx.info("���͉f���r�b�g���[�g: %d kbps", (int)srcBitrate);

    if (setting_.isAutoBitrate()) {
      ctx.info("�ڕW�f���r�b�g���[�g: %d kbps",
        (int)setting_.getBitrate().getTargetBitrate(fmt.format, srcBitrate));
    }

    // ������
    std::string args = makeEncoderArgs(
      setting_.getEncoder(),
      setting_.getEncoderPath(),
      setting_.getOptions(
				fmt.format, srcBitrate, false, pass_, 0, 0),
      fmt,
      setting_.getEncVideoFilePath(0, 0));

    ctx.info("[�G���R�[�_�J�n]");
    ctx.info(args.c_str());

    // x265�ŃC���^���[�X�̏ꍇ�̓t�B�[���h���[�h
    bool dstFieldMode =
      (setting_.getEncoder() == ENCODER_X265 && fmt.progressive == false);

    int bufsize = fmt.width * fmt.height * 3;
    encoder_->start(args, fmt, dstFieldMode, bufsize);
  }

  void onFrameDecoded(av::Frame& frame__) {
    // �t���[�����R�s�[���ăX���b�h�ɓn��
    thread_.put(std::unique_ptr<av::Frame>(new av::Frame(frame__)), 1);
  }

  void onFrameReceived(std::unique_ptr<av::Frame>&& frame)
  {
		// RFF�t���O����
		// PTS��inputFrame�ōĒ�`�����̂ŏC�����Ȃ��ł��̂܂ܓn��
		PICTURE_TYPE pic = getPictureTypeFromAVFrame((*frame)());
		//printf("%s\n", PictureTypeString(pic));
		rffExtractor_.inputFrame(*encoder_, std::move(frame), pic);

    //encoder_.inputFrame(*frame);
  }

  void onAudioPacket(AVPacket& packet)
  {
		if (pass_ <= 1) { // 2�p�X�ڂ͏o�͂��Ȃ�
			int audioIdx = audioMap_[packet.stream_index];
			if (audioIdx >= 0) {
				audioFiles_[audioIdx]->inputFrame(packet);
			}
		}
  }
};

class AMTMuxder : public AMTObject {
public:
	AMTMuxder(
		AMTContext&ctx,
		const TranscoderSetting& setting,
		const StreamReformInfo& reformInfo)
		: AMTObject(ctx)
		, setting_(setting)
		, reformInfo_(reformInfo)
		, audioCache_(ctx, setting.getAudioFilePath(), reformInfo.getAudioFileOffsets(), 12, 4)
		, totalOutSize_(0)
	{ }

	void mux(int videoFileIndex) {
		int numEncoders = reformInfo_.getNumEncoders(videoFileIndex);
		if (numEncoders == 0) {
			return;
		}

		for (int i = 0; i < numEncoders; ++i) {
			// �����t�@�C�����쐬
			std::vector<std::string> audioFiles;
			const FileAudioFrameList& fileFrameList =
				reformInfo_.getFileAudioFrameList(i, videoFileIndex);
			for (int a = 0; a < (int)fileFrameList.size(); ++a) {
				const std::vector<int>& frameList = fileFrameList[a];
				if (frameList.size() > 0) {
					std::string filepath = setting_.getIntAudioFilePath(videoFileIndex, i, a);
					File file(filepath, "wb");
					for (int frameIndex : frameList) {
						file.write(audioCache_[frameIndex]);
					}
					audioFiles.push_back(filepath);
				}
			}

			// �^�C���R�[�h�𖄂ߍ��ޕK�v�����邩
			bool needTimecode = reformInfo_.isVFR() ||
				(reformInfo_.hasRFF() && setting_.isPulldownEnabled());
			ctx.setCounter("timecode", (int)needTimecode);

			// Mux
			int outFileIndex = reformInfo_.getOutFileIndex(i, videoFileIndex);
			std::string encVideoFile = setting_.getEncVideoFilePath(videoFileIndex, i);
			std::string outFilePath = needTimecode
				? setting_.getVfrTmpFilePath(outFileIndex)
				: setting_.getOutFilePath(outFileIndex);
			std::string args = makeMuxerArgs(
				setting_.getMuxerPath(), encVideoFile,
				reformInfo_.getFormat(i, videoFileIndex).videoFormat,
				audioFiles, outFilePath);
			ctx.info("[Mux�J�n]");
			ctx.info(args.c_str());

			{
				MySubProcess muxer(args);
				int ret = muxer.join();
				if (ret != 0) {
					THROWF(RuntimeException, "mux failed (muxer exit code: %d)", ret);
				}
			}

			if (needTimecode) {
				std::string outWithTimeFilePath = setting_.getOutFilePath(outFileIndex);
				std::string encTimecodeFile = setting_.getEncTimecodeFilePath(videoFileIndex, i);
				auto timebase = reformInfo_.getTimebase(i, videoFileIndex);
				{ // �^�C���R�[�h�t�@�C���𐶐�
					std::ostringstream ss;
					ss << "# timecode format v2" << std::endl;
					const auto& timecode = reformInfo_.getTimecode(i, videoFileIndex);
					for (int64_t pts : timecode) {
						double dpts = (double)pts * timebase.second / timebase.first * 1000.0;
						ss << std::fixed << std::setprecision(2) << dpts << std::endl;
					}
					std::string str = ss.str();
					MemoryChunk mc(reinterpret_cast<uint8_t*>(const_cast<char*>(str.data())), str.size());
					File file(encTimecodeFile, "w");
					file.write(mc);
				}
				std::string args = makeTimelineEditorArgs(
					setting_.getTimelineEditorPath(), outFilePath, outWithTimeFilePath, encTimecodeFile, timebase);
				ctx.info("[�^�C���R�[�h���ߍ��݊J�n]");
				ctx.info(args.c_str());
				{
					MySubProcess timelineeditor(args);
					int ret = timelineeditor.join();
					if (ret != 0) {
						THROWF(RuntimeException, "timelineeditor failed (exit code: %d)", ret);
					}
				}
			}

			{ // �o�̓T�C�Y�擾
				File outfile(setting_.getOutFilePath(outFileIndex), "rb");
				totalOutSize_ += outfile.size();
			}
		}
	}

	int64_t getTotalOutSize() const {
		return totalOutSize_;
	}

private:
	class MySubProcess : public EventBaseSubProcess {
	public:
		MySubProcess(const std::string& args) : EventBaseSubProcess(args) { }
	protected:
		virtual void onOut(bool isErr, MemoryChunk mc) {
			// ����̓}���`�X���b�h�ŌĂ΂��̒���
			fwrite(mc.data, mc.length, 1, isErr ? stderr : stdout);
			fflush(isErr ? stderr : stdout);
		}
	};

	const TranscoderSetting& setting_;
	const StreamReformInfo& reformInfo_;

	PacketCache audioCache_;
	int64_t totalOutSize_;
};

class AMTSimpleMuxder : public AMTObject {
public:
	AMTSimpleMuxder(
		AMTContext&ctx,
		const TranscoderSetting& setting)
		: AMTObject(ctx)
		, setting_(setting)
		, totalOutSize_(0)
	{ }

	void mux(VideoFormat videoFormat, int audioCount) {
			// Mux
		std::vector<std::string> audioFiles;
		for (int i = 0; i < audioCount; ++i) {
			audioFiles.push_back(setting_.getIntAudioFilePath(0, 0, i));
		}
		std::string encVideoFile = setting_.getEncVideoFilePath(0, 0);
		std::string outFilePath = setting_.getOutFilePath(0);
		std::string args = makeMuxerArgs(
			setting_.getMuxerPath(), encVideoFile, videoFormat, audioFiles, outFilePath);
		ctx.info("[Mux�J�n]");
		ctx.info(args.c_str());

		{
			MySubProcess muxer(args);
			int ret = muxer.join();
			if (ret != 0) {
				THROWF(RuntimeException, "mux failed (muxer exit code: %d)", ret);
			}
		}

		{ // �o�̓T�C�Y�擾
			File outfile(setting_.getOutFilePath(0), "rb");
			totalOutSize_ += outfile.size();
		}
	}

	int64_t getTotalOutSize() const {
		return totalOutSize_;
	}

private:
	class MySubProcess : public EventBaseSubProcess {
	public:
		MySubProcess(const std::string& args) : EventBaseSubProcess(args) { }
	protected:
		virtual void onOut(bool isErr, MemoryChunk mc) {
			// ����̓}���`�X���b�h�ŌĂ΂��̒���
			fwrite(mc.data, mc.length, 1, isErr ? stderr : stdout);
			fflush(isErr ? stderr : stdout);
		}
	};

	const TranscoderSetting& setting_;
	int64_t totalOutSize_;
};

static std::vector<char> toUTF8String(const std::string str) {
	if (str.size() == 0) {
		return std::vector<char>();
	}
	int intlen = (int)str.size() * 2;
	auto wc = std::unique_ptr<wchar_t[]>(new wchar_t[intlen]);
	intlen = MultiByteToWideChar(CP_ACP, 0, str.c_str(), (int)str.size(), wc.get(), intlen);
	if (intlen == 0) {
		THROW(RuntimeException, "MultiByteToWideChar failed");
	}
	int dstlen = WideCharToMultiByte(CP_UTF8, 0, wc.get(), intlen, NULL, 0, NULL, NULL);
	if (dstlen == 0) {
		THROW(RuntimeException, "MultiByteToWideChar failed");
	}
	std::vector<char> ret(dstlen);
	WideCharToMultiByte(CP_UTF8, 0, wc.get(), intlen, ret.data(), (int)ret.size(), NULL, NULL);
	return ret;
}

static std::string toJsonString(const std::string str) {
	if (str.size() == 0) {
		return str;
	}
	std::vector<char> utf8 = toUTF8String(str);
	std::vector<char> ret;
	for (char c : utf8) {
		switch (c) {
		case '\"':
			ret.push_back('\\');
			ret.push_back('\"');
			break;
		case '\\':
			ret.push_back('\\');
			ret.push_back('\\');
			break;
		case '/':
			ret.push_back('\\');
			ret.push_back('/');
			break;
		case '\b':
			ret.push_back('\\');
			ret.push_back('b');
			break;
		case '\f':
			ret.push_back('\\');
			ret.push_back('f');
			break;
		case '\n':
			ret.push_back('\\');
			ret.push_back('n');
			break;
		case '\r':
			ret.push_back('\\');
			ret.push_back('r');
			break;
		case '\t':
			ret.push_back('\\');
			ret.push_back('t');
			break;
		default:
			ret.push_back(c);
		}
	}
	return std::string(ret.begin(), ret.end());
}

static void transcodeMain(AMTContext& ctx, const TranscoderSetting& setting)
{
	setting.dump();

	auto splitter = std::unique_ptr<AMTSplitter>(new AMTSplitter(ctx, setting));
	if (setting.getServiceId() > 0) {
		splitter->setServiceId(setting.getServiceId());
	}
	StreamReformInfo reformInfo = splitter->split();
	int64_t totalIntVideoSize = splitter->getTotalIntVideoSize();
	int64_t srcFileSize = splitter->getSrcFileSize();
	splitter = nullptr;

	if (setting.isDumpStreamInfo()) {
		reformInfo.serialize(setting.getStreamInfoPath());
	}

	reformInfo.prepareEncode();

	auto encoder = std::unique_ptr<AMTVideoEncoder>(new AMTVideoEncoder(ctx, setting, reformInfo));
  std::vector<EncodeFileInfo> bitrateInfo;
	for (int i = 0; i < reformInfo.getNumVideoFile(); ++i) {
    if (reformInfo.getNumEncoders(i) == 0) {
      ctx.warn("numEncoders == 0 ...");
    }
    else {
      auto efi = encoder->peform(i);
      bitrateInfo.insert(bitrateInfo.end(), efi.begin(), efi.end());
    }
	}
	encoder = nullptr;

	if (setting.getEncoder() == ENCODER_FFMPEG) {
		return;
	}

	auto audioDiffInfo = reformInfo.prepareMux();
	audioDiffInfo.printAudioPtsDiff(ctx);

  auto muxer = std::unique_ptr<AMTMuxder>(new AMTMuxder(ctx, setting, reformInfo));
  for (int i = 0; i < reformInfo.getNumVideoFile(); ++i) {
    muxer->mux(i);
  }
	int64_t totalOutSize = muxer->getTotalOutSize();
  muxer = nullptr;

	// �o�͌��ʂ�\��
	ctx.info("����");
	reformInfo.printOutputMapping([&](int index) { return setting.getOutFilePath(index); });

	// �o�͌���JSON�o��
	if (setting.getOutInfoJsonPath().size() > 0) {
		std::ostringstream ss;
		ss << "{ \"srcpath\": \"" << toJsonString(setting.getSrcFilePath()) << "\", ";
		ss << "\"outpath\": [";
		for (int i = 0; i < reformInfo.getNumOutFiles(); ++i) {
			if (i > 0) {
				ss << ", ";
			}
			ss << "\"" << toJsonString(setting.getOutFilePath(i)) << "\"";
		}
    ss << "], ";
    ss << "\"bitrate\": [";
    for (int i = 0; i < (int)bitrateInfo.size(); ++i) {
      auto info = bitrateInfo[i];
      if (i > 0) {
        ss << ", ";
      }
      ss << "{ \"src\": " << (int)info.srcBitrate
        << ", \"tgt1st\": " << (int)info.targetBitrate << "}";
    }
    ss << "], ";
		ss << "\"srcfilesize\": " << srcFileSize << ", ";
		ss << "\"intvideofilesize\": " << totalIntVideoSize << ", ";
		ss << "\"outfilesize\": " << totalOutSize << ", ";
		auto duration = reformInfo.getInOutDuration();
		ss << "\"srcduration\": " << std::fixed << std::setprecision(3)
			 << ((double)duration.first / MPEG_CLOCK_HZ) << ", ";
		ss << "\"outduration\": " << std::fixed << std::setprecision(3)
			 << ((double)duration.second / MPEG_CLOCK_HZ) << ", ";
		ss << "\"audiodiff\": ";
		audioDiffInfo.printToJson(ss);
		for (const auto& pair : ctx.getCounter()) {
			ss << ", \"" << pair.first << "\": " << pair.second;
		}
		ss << " }";

		std::string str = ss.str();
		MemoryChunk mc(reinterpret_cast<uint8_t*>(const_cast<char*>(str.data())), str.size());
		File file(setting.getOutInfoJsonPath(), "w");
		file.write(mc);
	}
}

static void transcodeSimpleMain(AMTContext& ctx, const TranscoderSetting& setting)
{
	if (ends_with(setting.getSrcFilePath(), ".ts")) {
		ctx.warn("��ʃt�@�C�����[�h�ł�TS�t�@�C���̏����͔񐄏��ł�");
	}

	auto encoder = std::unique_ptr<AMTSimpleVideoEncoder>(new AMTSimpleVideoEncoder(ctx, setting));
	encoder->encode();
	int audioCount = encoder->getAudioCount();
	int64_t srcFileSize = encoder->getSrcFileSize();
	VideoFormat videoFormat = encoder->getVideoFormat();
	encoder = nullptr;

	auto muxer = std::unique_ptr<AMTSimpleMuxder>(new AMTSimpleMuxder(ctx, setting));
	muxer->mux(videoFormat, audioCount);
	int64_t totalOutSize = muxer->getTotalOutSize();
	muxer = nullptr;

	// �o�͌��ʂ�\��
	ctx.info("����");
	if (setting.getOutInfoJsonPath().size() > 0) {
		std::ostringstream ss;
		ss << "{ \"srcpath\": \"" << toJsonString(setting.getSrcFilePath()) << "\", ";
		ss << "\"outpath\": [";
		ss << "\"" << toJsonString(setting.getOutFilePath(0)) << "\"";
		ss << "], ";
		ss << "\"srcfilesize\": " << srcFileSize << ", ";
		ss << "\"outfilesize\": " << totalOutSize;
		ss << " }";

		std::string str = ss.str();
		MemoryChunk mc(reinterpret_cast<uint8_t*>(const_cast<char*>(str.data())), str.size());
		File file(setting.getOutInfoJsonPath(), "w");
		file.write(mc);
	}
}

