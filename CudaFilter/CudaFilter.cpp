
#pragma warning(disable : 4819)

#define _CUDA_FILTER_EXPORT_
#include "CudaFilter.h"

#include <vector>
#include <deque>
#include <algorithm>

#include <Windows.h>
#undef max
#undef min

#include "cuda_runtime.h"

extern "C" {
#include <libavutil/imgutils.h>
}
#pragma comment(lib, "avutil.lib")

#include "kernel.h"

class TNRFilter {
public:
	TNRFilter(int temporalDistance, int threshold, int batchSize, int interlaced)
		: frame_info_()
	{
		temporalWidth_ = temporalDistance * 2 + 1;
		threshold_ = threshold;
		batchSize_ = batchSize;
		interlaced_ = interlaced;

		if (temporalWidth_ > TNR_MAX_WIDTH) {
			THROWF("時間軸距離が大きすぎます %d", temporalDistance);
		}
		if (batchSize_ > TNR_MAX_BATCH) {
			THROWF("バッチサイズが大きすぎます %d", batchSize_);
		}

		for (int i = 0; i < temporalWidth_; ++i) {
			kernel[i] = 1;
		}
	}
	~TNRFilter() {
		clear();
	}
	void sendFrame(AVFrame* frame)
	{
		if (out_queue_.size() > 0) {
			THROW("recvFrameして全て取り出してください");
		}
		if (frame_buffer_.size() == 0) {
			init(frame);
		}
		
		checkInputFrame(frame, "input");

		YV12Ptr ptr = frame_buffer_.back();
		frame_buffer_.pop_back();
		toGPU(&ptr, frame);
		frames_.push_back(ptr);

		int nframebuf = temporalWidth_ + batchSize_ - 1;
		int half = temporalWidth_ / 2;

		if (dup_first_ + frames_.size() == nframebuf) {
			// バッファが全て埋まったので処理する
			launchBatch(dup_first_, 0, batchSize_);
			dup_first_ = std::max(0, dup_first_ - batchSize_);
		}
	}
	int recvFrame(AVFrame* frame)
	{
		if (out_queue_.size() != 0) {
			toCPU(frame, out_queue_.front());
			out_queue_.pop_front();
			return 0;
		}
		return 1;
	}
	void finish()
	{
		int nframebuf = temporalWidth_ + batchSize_ - 1;
		int half = temporalWidth_ / 2;
		int remain = nframebuf - half;

		int dup_last = nframebuf - (dup_first_ + (int)frames_.size());
		for ( ; dup_last < remain; dup_last += batchSize_) {
			int validSize = std::min(batchSize_, remain - dup_last);
			launchBatch(dup_first_, dup_last, validSize);
			dup_first_ = std::max(0, dup_first_ - batchSize_);
		}

		while (frames_.size() > 0) {
			frame_buffer_.push_back(frames_.front());
			frames_.pop_front();
		}
	}

private:
	// パラメータ
	int temporalWidth_;
	int threshold_;
	int batchSize_;
	int interlaced_;

	float kernel[TNR_MAX_WIDTH];

	// 画像情報
	AVFrame frame_info_;
	int depth_;

	// データ
	std::vector<YV12Ptr> frame_buffer_;
	std::vector<OutBufPtr> out_buffer_;
	std::deque<YV12Ptr> frames_;
	std::deque<OutBufPtr*> out_queue_;

	int dup_first_;

	void init(AVFrame* frame)
	{
		clear();

		av_frame_copy_props(&frame_info_, frame);

		frame_info_.format = frame->format;
		int w = frame_info_.width = frame->width;
		int h = frame_info_.height = frame->height;
		int lsY = frame_info_.linesize[0] = frame->linesize[0];
		int lsU = frame_info_.linesize[1] = frame->linesize[1];
		int lsV = frame_info_.linesize[2] = frame->linesize[2];

		switch (frame_info_.format) {
		case AV_PIX_FMT_YUV420P:
		case AV_PIX_FMT_YUV420P10LE:
		case AV_PIX_FMT_YUV420P12LE:
		case AV_PIX_FMT_YUV420P14LE:
		case AV_PIX_FMT_YUV420P16LE:
			break;
		default:
			THROW("未対応フォーマットです");
		}

		const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get((AVPixelFormat)(frame_info_.format));
		depth_ = desc->comp[0].depth;
		dup_first_ = temporalWidth_ / 2;

		// メモリ確保
		int nframebuf = temporalWidth_ + batchSize_ - 1;
		int noutbuf = batchSize_;

		int frame_size = h * lsY + (h / 2) * (lsU +lsV);
		int out_size = frame_size + h * lsY * 2;

		for (int i = 0; i < nframebuf; ++i) {
			YV12Ptr ptr = {};
			CUDA_CHECK(cudaMalloc(&ptr.Y, frame_size));
			ptr.U = &ptr.Y[h * lsY];
			ptr.V = &ptr.U[(h / 2) * lsU];
			frame_buffer_.push_back(ptr);
		}

		for (int i = 0; i < noutbuf; ++i) {
			OutBufPtr ptr = {};
			CUDA_CHECK(cudaMalloc(&ptr.Y, out_size));
			ptr.U = &ptr.Y[h * lsY];
			ptr.V = &ptr.U[(h / 2) * lsU];
			ptr.iU = &ptr.V[(h / 2) * lsV];
			ptr.iV = &ptr.iU[h * lsY];;
			out_buffer_.push_back(ptr);
		}
	}

	void clear() {
		for (auto ptr : frame_buffer_) {
			CUDA_CHECK(cudaFree(ptr.Y));
		}
		for (auto ptr : out_buffer_) {
			CUDA_CHECK(cudaFree(ptr.Y));
		}
		frame_buffer_.clear();
		out_buffer_.clear();
		out_queue_.clear();
	}

	void checkInputFrame(AVFrame* frame, const char* name)
	{
#define CHECK(test) if(!(test)) THROWF("Frame %s propery changed %s", name, #test)

		CHECK(frame->width == frame_info_.width);
		CHECK(frame->height == frame_info_.height);
		CHECK(frame->format == frame_info_.format);
		CHECK(frame->linesize[0] == frame_info_.linesize[0]);
		CHECK(frame->linesize[1] == frame_info_.linesize[1]);
		CHECK(frame->linesize[2] == frame_info_.linesize[2]);

#undef CHECK
	}

	void launchBatch(int dup_first, int dup_last, int num_out)
	{
		int nframebuf = temporalWidth_ + batchSize_ - 1;
		std::vector<YV12Ptr> frames(nframebuf);
		int fidx = 0;
		for (int i = 0; i < dup_first; ++i) {
			frames[fidx++] = frames_.front();
		}
		for (int i = 0; i < (int)frames_.size(); ++i) {
			frames[fidx++] = frames_[i];
		}
		for (int i = 0; i < dup_last; ++i) {
			frames[fidx++] = frames_.back();
		}

		int scaledThresh = threshold_ << (depth_ - 8);

		cudaTemporalNRFilter(
			frame_info_.width, frame_info_.height, depth_, interlaced_ != 0,
			temporalWidth_, batchSize_, threshold_,
			frame_info_.linesize[0], frame_info_.linesize[1], frame_info_.linesize[2], 
			kernel, frames.data(), out_buffer_.data());

		for (int i = 0; i < num_out; ++i) {
			if (i >= dup_first) {
				frame_buffer_.push_back(frames_.front());
				frames_.pop_front();
			}
			out_queue_.push_back(&out_buffer_[i]);
		}
	}

	void toCPU(AVFrame* dst, OutBufPtr* src)
	{
		av_frame_unref(dst);

		// メモリサイズに関する情報をコピー
		dst->format = frame_info_.format;
		dst->width = frame_info_.width;
		dst->height = frame_info_.height;

		// メモリ確保
		if (av_frame_get_buffer(dst, 64) != 0) {
			THROW("failed to allocate frame buffer");
		}

		// linesizeが同じかチェック
		checkInputFrame(dst, "dst");

		int h = frame_info_.height;
		int szY = h * frame_info_.linesize[0];
		int szU = h / 2 * frame_info_.linesize[1];
		int szV = h / 2 * frame_info_.linesize[2];

		CUDA_CHECK(cudaMemcpy(dst->data[0], src->Y, szY, cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(dst->data[1], src->U, szU, cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(dst->data[2], src->V, szV, cudaMemcpyDeviceToHost));
	}

	void toGPU(YV12Ptr* dst, AVFrame* src)
	{
		int h = frame_info_.height;
		int szY = h * frame_info_.linesize[0];
		int szU = h / 2 * frame_info_.linesize[1];
		int szV = h / 2 * frame_info_.linesize[2];

		CUDA_CHECK(cudaMemcpy(dst->Y, src->data[0], szY, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(dst->U, src->data[1], szU, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(dst->V, src->data[2], szV, cudaMemcpyHostToDevice));
	}
};

LONG FilterException(struct _EXCEPTION_POINTERS * ExceptionInfo) {
	return UnhandledExceptionFilter(ExceptionInfo);
}

CudaTNRFilter cudaTNRCreateInternal(int temporalDistance, int threshold, int batchSize, int interlaced) {
	return new TNRFilter(temporalDistance, threshold, batchSize, interlaced);
}

EXPORT CudaTNRFilter cudaTNRCreate(int temporalDistance, int threshold, int batchSize, int interlaced)
{
	__try {
		return cudaTNRCreateInternal(temporalDistance, threshold, batchSize, interlaced);
	}
	__except (FilterException(GetExceptionInformation())) { }
	return NULL;
}

EXPORT int cudaTNRDelete(CudaTNRFilter * filter)
{
	__try {
		delete ((TNRFilter*)(*filter));
		*filter = NULL;
		return 0;
	}
	__except (FilterException(GetExceptionInformation())) { }
	return -1;
}

EXPORT int cudaTNRSendFrame(CudaTNRFilter filter, AVFrame * frame)
{
	__try {
		((TNRFilter*)filter)->sendFrame(frame);
		return 0;
	}
	__except (FilterException(GetExceptionInformation())) { }
	return -1;
}

EXPORT int cudaTNRRecvFrame(CudaTNRFilter filter, AVFrame * frame)
{
	__try {
		return ((TNRFilter*)filter)->recvFrame(frame);
	}
	__except (FilterException(GetExceptionInformation())) { }
	return -1;
}

EXPORT int cudaTNRFinish(CudaTNRFilter filter)
{
	__try {
		((TNRFilter*)filter)->finish();
		return 0;
	}
	__except (FilterException(GetExceptionInformation())) { }
	return -1;
}
