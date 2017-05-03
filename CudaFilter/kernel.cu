
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

struct TemporalNRParam {
	YV12Ptr frames[TNR_MAX_WIDTH + TNR_MAX_BATCH];
	OutBufPtr outbufs[TNR_MAX_BATCH];
	float kernel[TNR_MAX_WIDTH];
};

__constant__ TemporalNRParam tnr_param;

static int nblocks(int n, int width) {
	return (n + width - 1) / width;
}

template <typename VT>
__device__ int calcDiff(VT a, VT b) {
	int x = ((int)a.x - (int)b.x) * 2; // Yの重みを大きくする
	int y = (int)a.y - (int)b.y;
	int z = (int)a.z - (int)b.z;
	return
		((x >= 0) ? x : -x) +
		((y >= 0) ? y : -y) +
		((z >= 0) ? z : -z);
}

template <typename T, typename VT>
__global__ void klTemporalNRFilter(
	int nframes, int mid,
	int width, int height, int depth, bool interlaced,
	int lsY, int lsU, int lsV, // バイト単位ではなく要素単位
	int temporalWidth, int batchSize, int threshold)
{
	int b = threadIdx.y;
	int lx = threadIdx.x;
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = blockIdx.y;

	// [nframes][32]
	extern __shared__ void* s__[];
	VT (*pixel_cache)[32] = (VT(*)[32])s__;

	// pixel_cacheにデータを入れる
	int cy = interlaced ? (((y >> 1) & ~1) | (y & 1)) : (y >> 1);
	int cx = x >> 1;
	int offY = x + lsY * y;
	int offU = cx + lsU * cy;
	int offV = cx + lsV * cy;
	for (int i = b; i < nframes; i += blockDim.y) {
		T* Y = (T*)tnr_param.frames[i].Y;
		T* U = (T*)tnr_param.frames[i].U;
		T* V = (T*)tnr_param.frames[i].V;
		VT yuv = { Y[offY], U[offU], V[offV] };
		pixel_cache[i][lx] = yuv;
	}

	__syncthreads();

	VT center = pixel_cache[b + mid][lx];

	// 重み合計を計算
	float sumKernel = 0.0f;
	for (int i = 0; i < temporalWidth; ++i) {
		VT ref = pixel_cache[b + i][lx];
		int diff = calcDiff(center, ref);
		if (diff <= threshold) {
			sumKernel += tnr_param.kernel[i];
		}
	}

	float factor = 1.f / sumKernel;

	// ピクセル値を算出
	float dY = 0.5f;
	float dU = 0.5f;
	float dV = 0.5f;
	for (int i = 0; i < temporalWidth; ++i) {
		VT ref = pixel_cache[b + i][lx];
		int diff = calcDiff(center, ref);
		if (diff <= threshold) {
			float coef = tnr_param.kernel[i] * factor;
			dY += coef * ref.x;
			dU += coef * ref.y;
			dV += coef * ref.z;
		}
	}

	// YUV=4:4:4で一旦出力する
	T* outY = (T*)tnr_param.outbufs[b].Y;
	T* outU = (T*)tnr_param.outbufs[b].iU;
	T* outV = (T*)tnr_param.outbufs[b].iV;
	outY[offY] = (T)dY;
	outU[offY] = (T)dU;
	outV[offY] = (T)dV;
}

__device__ int g_test;

template <typename T>
__global__ void klConvert444to420(
	int width, int height, bool interlaced,
	int lsY, int lsU, int lsV, // バイト単位ではなく要素単位
	int batchSize)
{
	// x,yは420UV上での座標
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int yx = x * 2;
	int yy = (interlaced ? (((y&~1) << 1) | (y & 1)) : (y * 2));

	if (yx < width && yy < height) {
		int offY = yx + lsY * yy;
		int offYN = offY + lsY * (interlaced ? 2 : 1);
		int offU = x + lsU * y;
		int offV = x + lsV * y;

		for (int b = 0; b < batchSize; ++b) {
			T* iU = (T*)tnr_param.outbufs[b].iU;
			T* iV = (T*)tnr_param.outbufs[b].iV;
#if 0 // CPU版と合わせる場合
			int u = iU[offY];
			int v = iV[offY];
#else
			int u = ((int)iU[offY] + (int)iU[offY + 1] + (int)iU[offYN] + (int)iU[offYN + 1] + 2) >> 2;
			int v = ((int)iV[offY] + (int)iV[offY + 1] + (int)iV[offYN] + (int)iV[offYN + 1] + 2) >> 2;
#endif
			T* U = (T*)tnr_param.outbufs[b].U;
			T* V = (T*)tnr_param.outbufs[b].V;
			U[offU] = u;
			V[offV] = v;
		}
	}
}

void cudaTemporalNRFilter(
	int width, int height, int depth, bool interlaced,
	int temporalWidth, int batchSize, int threshold,
	int lsY, int lsU, int lsV,
	float* kernel, YV12Ptr* frames, OutBufPtr* outbufs)
{
	int nframes = temporalWidth + batchSize - 1;
	int mid = temporalWidth / 2;

	// パラメータをコピー
	TemporalNRParam param = { 0 };
	memcpy(param.frames, frames, nframes*sizeof(frames[0]));
	memcpy(param.outbufs, outbufs, batchSize*sizeof(outbufs[0]));
	memcpy(param.kernel, kernel, temporalWidth*sizeof(kernel[0]));
	CUDA_CHECK(cudaMemcpyToSymbol(tnr_param, &param, sizeof(param), 0));

	dim3 threadsF(32, batchSize);
	dim3 blocksF(nblocks(width, threadsF.x), height);
	dim3 threadsC(32, 16);
	dim3 blocksC(nblocks(width / 2, threadsC.x), nblocks(height / 2, threadsC.y));

#if 0
	for (int i = 0; i < batchSize; ++i) {
		CUDA_CHECK(cudaMemcpy(outbufs[i].Y, frames[i + mid].Y, height * lsY, cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(outbufs[i].U, frames[i + mid].U, height / 2 * lsU, cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(outbufs[i].V, frames[i + mid].V, height / 2 * lsV, cudaMemcpyDeviceToDevice));
	}
#else
	if (depth <= 8) {
		int shared_size = threadsF.x*nframes*sizeof(uchar4);
		klTemporalNRFilter<uint8_t, uchar4><<<blocksF, threadsF, shared_size>>>(
			nframes, mid, width, height, depth, interlaced,
			lsY, lsU, lsV,
			temporalWidth, batchSize, threshold);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		klConvert444to420<uint8_t><<<blocksC, threadsC >>>(
			width, height, interlaced,
			lsY, lsU, lsV, 
			batchSize);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else {
		int shared_size = threadsF.x*nframes*sizeof(ushort4);
		klTemporalNRFilter<uint16_t, ushort4><<<blocksF, threadsF, shared_size>>>(
			nframes, mid, width, height, depth, interlaced,
			lsY / sizeof(uint16_t), lsU / sizeof(uint16_t), lsV / sizeof(uint16_t),
			temporalWidth, batchSize, threshold);
		CUDA_CHECK(cudaDeviceSynchronize());
		klConvert444to420<uint16_t><<<blocksC, threadsC >>>(
			width, height, interlaced,
			lsY / sizeof(uint16_t), lsU / sizeof(uint16_t), lsV / sizeof(uint16_t),
			batchSize);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
#endif
}
