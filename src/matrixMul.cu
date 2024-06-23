#include "matrixMul.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matrixMulKernelNaive(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < cWidth && y < cHeight) {
		float sum = 0;
		for (int i = 0; i < aWidth; i++) {
			sum += a[y * aWidth + i] * b[i * cWidth + x];
		}
		c[y * cWidth + x] = sum;
	}
}

void matrixMulNaive(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
	dim3 blockSize(32, 32);
	dim3 gridSize((cWidth + blockSize.x - 1) / blockSize.x,
		(cHeight + blockSize.y - 1) / blockSize.y);
	matrixMulKernelNaive<<<gridSize, blockSize>>>(c, a, b, cWidth, cHeight, aWidth);
}

template <int TILE_SIZE>
__global__ void matrixMulKernelSTiled(float* c, const float* a, const float* b, int cWidth, int aWidth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	assert(blockDim.x == TILE_SIZE && blockDim.y == TILE_SIZE);

	__shared__ float aShared[TILE_SIZE][TILE_SIZE];
	__shared__ float bShared[TILE_SIZE][TILE_SIZE];

	float sum = 0;

	for (int i = 0; i < aWidth; i += TILE_SIZE) {
		aShared[threadIdx.y][threadIdx.x] = a[y * aWidth + (i + threadIdx.x)];
		bShared[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * cWidth + x];
		__syncthreads();

		for (int j = 0; j < TILE_SIZE; j++) {
			sum += aShared[threadIdx.y][j] * bShared[j][threadIdx.x];
		}
		__syncthreads();
	}

	c[y * cWidth + x] = sum;
}

void matrixMulSTiled(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
	const int TILE_SIZE = 16;
	assert(cWidth % TILE_SIZE == 0);
	assert(cHeight % TILE_SIZE == 0);
	assert(aWidth % TILE_SIZE == 0);
	dim3 blockSize(TILE_SIZE, TILE_SIZE);
	dim3 gridSize(cWidth / TILE_SIZE, cHeight / TILE_SIZE);
	matrixMulKernelSTiled<TILE_SIZE><<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
}

template <int TILE_SIZE, int BLOCK_SIZE>
__global__ void matrixMulKernelSTTiled(float* c, const float* a, const float* b, int cWidth, int aWidth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	assert(blockDim.x == BLOCK_SIZE && blockDim.y == BLOCK_SIZE);

	__shared__ float4 aShared[BLOCK_SIZE * TILE_SIZE * BLOCK_SIZE];
	__shared__ float4 bShared[BLOCK_SIZE * TILE_SIZE * BLOCK_SIZE];

	float4 row0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 row1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 row2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 row3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	for (int i = 0; i < aWidth; i += BLOCK_SIZE * TILE_SIZE) {
		#pragma unroll
		for (int j = 0; j < TILE_SIZE; j++) {
			aShared[(threadIdx.y * TILE_SIZE + j) * BLOCK_SIZE + threadIdx.x] =
				*((const float4*)(a + ((y * TILE_SIZE + j) * aWidth + (i + threadIdx.x * TILE_SIZE))));
			bShared[(threadIdx.y * TILE_SIZE + j) * BLOCK_SIZE + threadIdx.x] =
				*((const float4*)(b + ((i + threadIdx.y * TILE_SIZE + j) * cWidth + (x * TILE_SIZE))));
		}
		__syncthreads();

		float* aS = (float*)aShared + threadIdx.y * TILE_SIZE * BLOCK_SIZE * TILE_SIZE;
		float* bS = (float*)bShared + (threadIdx.x * TILE_SIZE);

        #pragma unroll
		for (int j = 0; j < BLOCK_SIZE * TILE_SIZE; j++) {
			float a0 = aS[0 * BLOCK_SIZE * TILE_SIZE + j];
			float a1 = aS[1 * BLOCK_SIZE * TILE_SIZE + j];
			float a2 = aS[2 * BLOCK_SIZE * TILE_SIZE + j];
			float a3 = aS[3 * BLOCK_SIZE * TILE_SIZE + j];
			float4 b = *((float4*)(bS + j * BLOCK_SIZE * TILE_SIZE));

			row0.x += a0 * b.x;
			row0.y += a0 * b.y;
			row0.z += a0 * b.z;
			row0.w += a0 * b.w;

			row1.x += a1 * b.x;
			row1.y += a1 * b.y;
			row1.z += a1 * b.z;
			row1.w += a1 * b.w;

			row2.x += a2 * b.x;
			row2.y += a2 * b.y;
			row2.z += a2 * b.z;
			row2.w += a2 * b.w;

			row3.x += a3 * b.x;
			row3.y += a3 * b.y;
			row3.z += a3 * b.z;
			row3.w += a3 * b.w;
		}
		__syncthreads();
	}

	c = (c + y * TILE_SIZE * cWidth) + x * TILE_SIZE;
	*((float4*)c) = row0;
	c += cWidth;
	*((float4*)c) = row1;
	c += cWidth;
	*((float4*)c) = row2;
	c += cWidth;
	*((float4*)c) = row3;
}

void matrixMulSTTiled(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
	const int TILE_SIZE = 4;
	const int BLOCK_SIZE = 16;
	assert(cWidth % TILE_SIZE == 0);
	assert(cHeight % TILE_SIZE == 0);
	assert(aWidth % TILE_SIZE == 0);
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 matPartSize(blockSize.x * TILE_SIZE, blockSize.y * TILE_SIZE);
	assert(cWidth % matPartSize.x == 0);
	assert(cHeight % matPartSize.y == 0);
	dim3 gridSize(cWidth / matPartSize.x, cHeight / matPartSize.y);
	matrixMulKernelSTTiled<TILE_SIZE, BLOCK_SIZE><<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
}

template <int WIDTH, int HEIGHT, int THREAD_COUNT>
__device__ void loadRect(float* dst, const float* src, int srcStride, int thread) {
	const int VEC_SIZE = sizeof(float4) / sizeof(float);
	assert(WIDTH % VEC_SIZE == 0);
	const int W = WIDTH / VEC_SIZE;
	assert(WIDTH / VEC_SIZE <= THREAD_COUNT);
	const int H = THREAD_COUNT / W;
	assert(THREAD_COUNT % W == 0);

	int ty = thread / W;
	int tx = thread - (ty * W);
	tx *= VEC_SIZE;
	src += tx;
	dst += tx;

    #pragma unroll
	for (int i = ty; i < HEIGHT; i += H) {
		float4 vec = *reinterpret_cast<const float4*>(src + i * srcStride);
		reinterpret_cast<float4*>(dst)[i * W] = vec;
	}
}

template <int SHARED_SIZE, int TILE_WIDTH, int TILE_HEIGHT, int GRID_WIDTH, int GRID_HEIGHT>
__global__ void matrixMulKernelSWTiled(float* c, const float* a, const float* b, int cWidth, int aWidth)
{
	assert(blockDim.x == TILE_WIDTH);
	assert(blockDim.y == GRID_WIDTH && blockDim.z == GRID_HEIGHT);
	const int THREAD_COUNT = TILE_WIDTH * GRID_WIDTH * GRID_HEIGHT;

	int warpThread = threadIdx.x;
	int tileX = threadIdx.y;
	int tileY = threadIdx.z;
	int thread = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
	int x = blockIdx.x * blockDim.y * TILE_WIDTH;
	int y = blockIdx.y * blockDim.z * TILE_HEIGHT;

	__shared__ float aS[GRID_HEIGHT * TILE_HEIGHT * SHARED_SIZE];
	__shared__ float bS[SHARED_SIZE * GRID_WIDTH * TILE_WIDTH];

	float accum[TILE_HEIGHT];
	for (int i = 0; i < TILE_HEIGHT; i++)
		accum[i] = 0;

	a += y * aWidth;
	b += x;

	for (int i = 0; i < aWidth; i += SHARED_SIZE) {
		loadRect<SHARED_SIZE, GRID_HEIGHT * TILE_HEIGHT, THREAD_COUNT>((float*)aS, a + i, aWidth, thread);
		loadRect<GRID_WIDTH * TILE_WIDTH, SHARED_SIZE, THREAD_COUNT>((float*)bS, b + i * cWidth, cWidth, thread);
		__syncthreads();

		float* a = aS + tileY * TILE_HEIGHT * SHARED_SIZE;
		float* b = bS + tileX * TILE_WIDTH + warpThread;
        #pragma unroll
		for (int k = 0; k < SHARED_SIZE; k++) {
			float bValue = b[k * GRID_WIDTH * TILE_WIDTH];
			#pragma unroll
			for (int j = 0; j < TILE_HEIGHT; j++) {
				accum[j] += a[j * SHARED_SIZE + k] * bValue;
			}
		}
		__syncthreads();
	}

	x += tileX * TILE_WIDTH;
	y += tileY * TILE_HEIGHT;
	c += x + warpThread;
	c += y * cWidth;
	#pragma unroll
	for (int i = 0; i < TILE_HEIGHT; i++)
		c[i * cWidth] = accum[i];
}

void matrixMulSWTiled(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
	const int WARP_SIZE = 32;
	const int SHARED_SIZE = 32;
	const int TILE_WIDTH = WARP_SIZE;
	const int TILE_HEIGHT = 64;
	const int GRID_WIDTH = 4;
	const int GRID_HEIGHT = 2;
	dim3 blockSize(WARP_SIZE, GRID_WIDTH, GRID_HEIGHT);
	dim3 accumSize(blockSize.y * TILE_WIDTH, blockSize.z * TILE_HEIGHT);
	assert(aWidth % SHARED_SIZE == 0);
	assert(cWidth % accumSize.x == 0);
	assert(cHeight % accumSize.y == 0);
	dim3 gridSize(cWidth / accumSize.x, cHeight / accumSize.y);
	matrixMulKernelSWTiled<SHARED_SIZE, TILE_WIDTH, TILE_HEIGHT, GRID_WIDTH, GRID_HEIGHT>
		<<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
}

template <int WIDTH, int HEIGHT, int THREAD_COUNT>
__device__ void loadRectT(float* dst, const float* src, int srcStride, int thread) {
	const int VEC_SIZE = sizeof(float4) / sizeof(float);
	assert(WIDTH % VEC_SIZE == 0);
	if (THREAD_COUNT == HEIGHT * 2) {
		int ty = thread >> 1;
		int tx = thread - ty * 2;
		const float4* srcRow = reinterpret_cast<const float4*>(src + ty * srcStride);
		dst += ty;

		#pragma unroll
		for (int x = tx; x < WIDTH / VEC_SIZE; x += 2) {
			float4 vec = srcRow[x];
			dst[(x * VEC_SIZE + 0) * HEIGHT] = vec.x;
			dst[(x * VEC_SIZE + 1) * HEIGHT] = vec.y;
			dst[(x * VEC_SIZE + 2) * HEIGHT] = vec.z;
			dst[(x * VEC_SIZE + 3) * HEIGHT] = vec.w;
		}
	}
	else {
		assert(THREAD_COUNT % 2 == 0);
		const int THREAD_BLOCK_HEIGHT = min(THREAD_COUNT / 2, HEIGHT);
		assert(THREAD_COUNT % THREAD_BLOCK_HEIGHT == 0);
		const int THREAD_BLOCK_WIDTH = THREAD_COUNT / THREAD_BLOCK_HEIGHT;
		assert((WIDTH / VEC_SIZE) % THREAD_BLOCK_WIDTH == 0);
		assert(HEIGHT % THREAD_BLOCK_HEIGHT == 0);

		int ty = thread / THREAD_BLOCK_WIDTH;
		int tx = thread - (ty * THREAD_BLOCK_WIDTH);

		#pragma unroll
		for (int y = ty; y < HEIGHT; y += THREAD_BLOCK_HEIGHT) {
			const float4* srcRow = reinterpret_cast<const float4*>(src + y * srcStride);
			float* dstCol = dst + y;
			#pragma unroll
			for (int x = tx; x < WIDTH / VEC_SIZE; x += THREAD_BLOCK_WIDTH) {
				float4 vec = srcRow[x];
				dstCol[(x * VEC_SIZE + 0) * HEIGHT] = vec.x;
				dstCol[(x * VEC_SIZE + 1) * HEIGHT] = vec.y;
				dstCol[(x * VEC_SIZE + 2) * HEIGHT] = vec.z;
				dstCol[(x * VEC_SIZE + 3) * HEIGHT] = vec.w;
			}
		}
	}
}

template <int SHARED_SIZE, int TILE_WIDTH, int TILE_HEIGHT, int GRID_WIDTH, int GRID_HEIGHT>
__global__ void matrixMulKernelFast(float* c, const float* a, const float* b, int cWidth, int aWidth)
{
	const int WARP_SIZE = 32;
	const int VEC_SIZE = sizeof(float4) / sizeof(float);
	const int ACCUM_WIDTH = GRID_WIDTH * TILE_WIDTH;
	const int ACCUM_HEIGHT = GRID_HEIGHT * TILE_HEIGHT;
	const int THREAD_COUNT = WARP_SIZE * GRID_WIDTH * GRID_HEIGHT;
	const int WARP_WIDTH = TILE_WIDTH / VEC_SIZE / 2;
	const int WARP_HEIGHT = TILE_HEIGHT / VEC_SIZE / 2;

	assert(blockDim.x == WARP_SIZE);
	assert(blockDim.y == GRID_WIDTH && blockDim.z == GRID_HEIGHT);
	assert(WARP_WIDTH * WARP_HEIGHT == WARP_SIZE);

	int warpThread = threadIdx.x;
	int tileX = threadIdx.y;
	int tileY = threadIdx.z;
	int threadX = warpThread % WARP_WIDTH;
	int threadY = warpThread / WARP_WIDTH;
	int thread = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
	int x = blockIdx.x * blockDim.y * TILE_WIDTH;
	int y = blockIdx.y * blockDim.z * TILE_HEIGHT;

	__shared__ float aS[SHARED_SIZE * ACCUM_HEIGHT];
	__shared__ float bS[SHARED_SIZE * ACCUM_WIDTH];

	float4 accum00[VEC_SIZE];
	float4 accum01[VEC_SIZE];
	float4 accum10[VEC_SIZE];
	float4 accum11[VEC_SIZE];
    #pragma unroll
	for (int i = 0; i < VEC_SIZE; i++) {
		accum00[i] = accum01[i] = accum10[i] = accum11[i] = make_float4(0.f, 0.f, 0.f, 0.f);
	}

	a += y * aWidth;
	b += x;

	for (int i = 0; i < aWidth; i += SHARED_SIZE) {
		loadRectT<SHARED_SIZE, GRID_HEIGHT * TILE_HEIGHT, THREAD_COUNT>((float*)aS, a + i, aWidth, thread);
		loadRect<GRID_WIDTH * TILE_WIDTH, SHARED_SIZE, THREAD_COUNT>((float*)bS, b + i * cWidth, cWidth, thread);
		__syncthreads();

		float4* b = reinterpret_cast<float4*>(bS + tileX * TILE_WIDTH) + threadX;
		float4* a = reinterpret_cast<float4*>(aS + tileY * TILE_HEIGHT) + threadY;
		const int bStride = ACCUM_WIDTH / VEC_SIZE;
		const int aStride = ACCUM_HEIGHT / VEC_SIZE;

        #pragma unroll
		for (int j = 0; j < SHARED_SIZE; j++) {
			float4 b0, b1;
			b0 = b[j * bStride];
			b1 = b[j * bStride + WARP_WIDTH];
			float4 a0, a1;
			a0 = a[j * aStride];
			a1 = a[j * aStride + WARP_HEIGHT];

			accum00[0].x += a0.x * b0.x; accum00[0].y += a0.x * b0.y; accum00[0].z += a0.x * b0.z; accum00[0].w += a0.x * b0.w;
			accum00[1].x += a0.y * b0.x; accum00[1].y += a0.y * b0.y; accum00[1].z += a0.y * b0.z; accum00[1].w += a0.y * b0.w;
			accum00[2].x += a0.z * b0.x; accum00[2].y += a0.z * b0.y; accum00[2].z += a0.z * b0.z; accum00[2].w += a0.z * b0.w;
			accum00[3].x += a0.w * b0.x; accum00[3].y += a0.w * b0.y; accum00[3].z += a0.w * b0.z; accum00[3].w += a0.w * b0.w;

			accum01[0].x += a0.x * b1.x; accum01[0].y += a0.x * b1.y; accum01[0].z += a0.x * b1.z; accum01[0].w += a0.x * b1.w;
			accum01[1].x += a0.y * b1.x; accum01[1].y += a0.y * b1.y; accum01[1].z += a0.y * b1.z; accum01[1].w += a0.y * b1.w;
			accum01[2].x += a0.z * b1.x; accum01[2].y += a0.z * b1.y; accum01[2].z += a0.z * b1.z; accum01[2].w += a0.z * b1.w;
			accum01[3].x += a0.w * b1.x; accum01[3].y += a0.w * b1.y; accum01[3].z += a0.w * b1.z; accum01[3].w += a0.w * b1.w;

			accum10[0].x += a1.x * b0.x; accum10[0].y += a1.x * b0.y; accum10[0].z += a1.x * b0.z; accum10[0].w += a1.x * b0.w;
			accum10[1].x += a1.y * b0.x; accum10[1].y += a1.y * b0.y; accum10[1].z += a1.y * b0.z; accum10[1].w += a1.y * b0.w;
			accum10[2].x += a1.z * b0.x; accum10[2].y += a1.z * b0.y; accum10[2].z += a1.z * b0.z; accum10[2].w += a1.z * b0.w;
			accum10[3].x += a1.w * b0.x; accum10[3].y += a1.w * b0.y; accum10[3].z += a1.w * b0.z; accum10[3].w += a1.w * b0.w;

			accum11[0].x += a1.x * b1.x; accum11[0].y += a1.x * b1.y; accum11[0].z += a1.x * b1.z; accum11[0].w += a1.x * b1.w;
			accum11[1].x += a1.y * b1.x; accum11[1].y += a1.y * b1.y; accum11[1].z += a1.y * b1.z; accum11[1].w += a1.y * b1.w;
			accum11[2].x += a1.z * b1.x; accum11[2].y += a1.z * b1.y; accum11[2].z += a1.z * b1.z; accum11[2].w += a1.z * b1.w;
			accum11[3].x += a1.w * b1.x; accum11[3].y += a1.w * b1.y; accum11[3].z += a1.w * b1.z; accum11[3].w += a1.w * b1.w;
		}
		__syncthreads();
	}

	x += tileX * TILE_WIDTH + threadX * VEC_SIZE;
	y += tileY * TILE_HEIGHT + threadY * VEC_SIZE;
#define STORE_OPT
#ifdef STORE_OPT
	c += y * cWidth + x;
    #pragma unroll
	for (int i = 0; i < VEC_SIZE; i++) {
		c[i * cWidth + 0] = accum00[i].x;
		c[i * cWidth + 1] = accum00[i].y;
		c[i * cWidth + 2] = accum00[i].z;
		c[i * cWidth + 3] = accum00[i].w;
	}
	for (int i = 0; i < VEC_SIZE; i++) {
		c[i * cWidth + VEC_SIZE * WARP_WIDTH + 0] = accum01[i].x;
		c[i * cWidth + VEC_SIZE * WARP_WIDTH + 1] = accum01[i].y;
		c[i * cWidth + VEC_SIZE * WARP_WIDTH + 2] = accum01[i].z;
		c[i * cWidth + VEC_SIZE * WARP_WIDTH + 3] = accum01[i].w;
	}
	for (int i = 0; i < VEC_SIZE; i++) {
		c[i * cWidth + VEC_SIZE * WARP_HEIGHT * cWidth + 0] = accum10[i].x;
		c[i * cWidth + VEC_SIZE * WARP_HEIGHT * cWidth + 1] = accum10[i].y;
		c[i * cWidth + VEC_SIZE * WARP_HEIGHT * cWidth + 2] = accum10[i].z;
		c[i * cWidth + VEC_SIZE * WARP_HEIGHT * cWidth + 3] = accum10[i].w;
	}
	for (int i = 0; i < VEC_SIZE; i++) {
		c[i * cWidth + VEC_SIZE * (WARP_HEIGHT * cWidth + WARP_WIDTH) + 0] = accum11[i].x;
		c[i * cWidth + VEC_SIZE * (WARP_HEIGHT * cWidth + WARP_WIDTH) + 1] = accum11[i].y;
		c[i * cWidth + VEC_SIZE * (WARP_HEIGHT * cWidth + WARP_WIDTH) + 2] = accum11[i].z;
		c[i * cWidth + VEC_SIZE * (WARP_HEIGHT * cWidth + WARP_WIDTH) + 3] = accum11[i].w;
	}
#else
	float4* out = reinterpret_cast<float4*>(c + y * cWidth + x);
	int outStride = cWidth >> 2;
	#pragma unroll
	for (int i = 0; i < vecSize; i++) {
		out[i * outStride] = accum00[i];
		out[i * outStride + warpWidth] = accum01[i];
		out[i * outStride + warpHeight * cWidth] = accum10[i];
		out[i * outStride + warpHeight * cWidth + warpWidth] = accum11[i];
	}
#endif
}

void matrixMulFast(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
	const int WARP_SIZE = 32;
	const int SHARED_SIZE = 32;
	const int TILE_WIDTH = 64;
	const int TILE_HEIGHT = 32;
	const int GRID_WIDTH = 2;
	const int GRID_HEIGHT = 2;
	dim3 blockSize(WARP_SIZE, GRID_WIDTH, GRID_HEIGHT);
	dim3 accumSize(GRID_WIDTH * TILE_WIDTH, GRID_HEIGHT * TILE_HEIGHT);
	assert(aWidth % SHARED_SIZE == 0);
	assert(cWidth % accumSize.x == 0);
	assert(cHeight % accumSize.y == 0);
	dim3 gridSize(cWidth / accumSize.x, cHeight / accumSize.y);
	matrixMulKernelFast<SHARED_SIZE, TILE_WIDTH, TILE_HEIGHT, GRID_WIDTH, GRID_HEIGHT>
		<<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
}

using KernelFunction = void(*)(float*, const float*, const float*, int, int, int);

float matrixMul(CudaBuffer<float>& cHost, const CudaBuffer<float>& aHost, const CudaBuffer<float>& bHost,
	            int cWidth, int cHeight, int aWidth, KernelFunction kernelFunc) {
	CudaBuffer<float> cDev(cHost.size(), cudaMemoryTypeDevice);
	CudaBuffer<float> aDev(aHost.size(), cudaMemoryTypeDevice);
	CudaBuffer<float> bDev(bHost.size(), cudaMemoryTypeDevice);
	aDev.copyFrom(aHost);
	bDev.copyFrom(bHost);

	CudaEvent start, stop;
	start.record();
	kernelFunc(cDev(), aDev(), bDev(), cWidth, cHeight, aWidth);
	checkCudaError(cudaGetLastError());
	stop.record();
	stop.synchronize();
	float elapsedTime = start.elapsedTime(stop);

	checkCudaError(cudaDeviceSynchronize());
	cHost.copyFrom(cDev);

	return elapsedTime;
}

void matrixMulRef(float* c, const float* a, const float* b,
	int cWidth, int cHeight, int aWidth, bool tiled = true) { // row-major order
	if (!tiled) {
		for (int y = 0; y < cHeight; y++) {
			for (int x = 0; x < cWidth; x++) {
				float sum = 0;
				for (int i = 0; i < aWidth; i++) {
					sum += a[y * aWidth + i] * b[i * cWidth + x];
				}
				c[y * cWidth + x] = sum;
			}
		}
	}
	else {
		const int TILE_SIZE = 128;
		float accum[TILE_SIZE][TILE_SIZE];
		for (int ty = 0; ty < cHeight; ty += TILE_SIZE) {
			for (int tx = 0; tx < cWidth; tx += TILE_SIZE) {
				int tileWidth = std::min(cWidth - tx, TILE_SIZE);
				int tileHeight = std::min(cHeight - ty, TILE_SIZE);
				for (int y = 0; y < tileHeight; y++) {
					for (int x = 0; x < tileWidth; x++) {
						accum[y][x] = 0;
					}
				}
				for (int i = 0; i < aWidth; i++) {
					const float* aPtr = a + ty * aWidth + i;
					const float* bPtr = b + i * cWidth + tx;
					for (int y = 0; y < tileHeight; y++) {
						float aValue = aPtr[y * aWidth];
						for (int x = 0; x < tileWidth; x++) {
							accum[y][x] += aValue * bPtr[x];
						}
					}
				}
				for (int y = 0; y < tileHeight; y++) {
					for (int x = 0; x < tileWidth; x++) {
						c[(ty + y) * cWidth + (tx + x)] = accum[y][x];
					}
				}
			}
		}
	}
}

void testMatrixMul(int cWidth, int cHeight, int aWidth,
	               KernelFunction kernelFunc, const std::string& kernelName = "") {
	assert(cWidth > 0 && cHeight > 0 && aWidth > 0);
	CudaBuffer<float> a(aWidth * cHeight, cudaMemoryTypeHost);
	CudaBuffer<float> b(cWidth * aWidth, cudaMemoryTypeHost);
	a.fillWithRandom();
	b.fillWithRandom();

	CudaBuffer<float> c(cWidth * cHeight, cudaMemoryTypeHost);
	float time = matrixMul(c, a, b, cWidth, cHeight, aWidth, kernelFunc);
	CudaBuffer<float> cRef(cWidth * cHeight, cudaMemoryTypeHost);
	matrixMulRef(cRef(), a(), b(), cWidth, cHeight, aWidth);

	bool pass = c.approxEqual(cRef);
	std::string name = "MatrixMul";
	name += kernelName + " (";
	name += std::to_string(cWidth) + "x";
	name += std::to_string(cHeight);
	name += ", " + std::to_string(aWidth) + ")";
	printTestItem(name, pass, time);
	checkCudaError(cudaDeviceReset());
}

void testMatrixMul() {
	checkCudaError(cudaSetDevice(0));

	testMatrixMul(1024, 1024, 1024, matrixMulNaive, "Naive");
	testMatrixMul(1024, 1024, 1024, matrixMulSTiled, "STiled");
	testMatrixMul(1024, 1024, 1024, matrixMulSTTiled, "STTiled");
	testMatrixMul(1024, 1024, 1024, matrixMulSWTiled, "SWTiled");
	testMatrixMul(1024, 1024, 1024, matrixMulFast, "Fast");
}
