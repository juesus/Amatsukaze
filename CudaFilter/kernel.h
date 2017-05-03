#pragma once

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>

#define THROW(message) \
  throw_exception_("Exception thrown at %s:%d\r\nMessage: " message, __FILE__, __LINE__)

#define THROWF(fmt, ...) \
  throw_exception_("Exception thrown at %s:%d\r\nMessage: " fmt, __FILE__, __LINE__, __VA_ARGS__)

static void throw_exception_(const char* fmt, ...)
{
	char buf[300];
	va_list arg;
	va_start(arg, fmt);
	vsnprintf_s(buf, sizeof(buf), fmt, arg);
	va_end(arg);
	printf(buf);
	throw buf;
}

#define CUDA_CHECK(call) \
		do { \
			cudaError_t err__ = call; \
			if (err__ != cudaSuccess) { \
				THROWF("[CUDA Error] %d: %s", err__, cudaGetErrorString(err__)); \
			} \
		} while (0)

enum { 
	TNR_MAX_WIDTH = 128,
	TNR_MAX_BATCH = 16
};

struct YV12Ptr {
	uint8_t *Y, *U, *V;
};

struct OutBufPtr {
	uint8_t *Y, *U, *V;
	uint8_t *iU, *iV;
};

void cudaTemporalNRFilter(
	int width, int height, int depth, bool interlaced,
	int temporalWidth, int batchSize, int threshold,
	int lsY, int lsU, int lsV,
	float* kernel, YV12Ptr* frames, OutBufPtr* outbufs);
