#pragma once

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <libavformat/avformat.h>

#ifdef _CUDA_FILTER_EXPORT_
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#pragma comment(lib, "CudaFilter.lib")
#endif

typedef void* CudaTNRFilter;

EXPORT CudaTNRFilter cudaTNRCreate(int temporalDistance, int threshold, int batchSize, int interlaced);

EXPORT int cudaTNRDelete(CudaTNRFilter* filter);

EXPORT int cudaTNRSendFrame(CudaTNRFilter filter, AVFrame* frame);

EXPORT int cudaTNRRecvFrame(CudaTNRFilter filter, AVFrame* frame);

EXPORT int cudaTNRFinish(CudaTNRFilter filter);

#ifdef __cplusplus
}
#endif /* __cplusplus */
