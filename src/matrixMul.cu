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

__global__ void matrixMulKernelSTiled(float* c, const float* a, const float* b, int cWidth, int aWidth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int tileSize = 16;
	assert(blockDim.x == tileSize && blockDim.y == tileSize);

	__shared__ float aShared[tileSize][tileSize];
	__shared__ float bShared[tileSize][tileSize];

	float sum = 0;

	for (int i = 0; i < aWidth; i += tileSize) {
		aShared[threadIdx.y][threadIdx.x] = a[y * aWidth + (i + threadIdx.x)];
		bShared[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * cWidth + x];
		__syncthreads();

		for (int j = 0; j < tileSize; j++) {
			sum += aShared[threadIdx.y][j] * bShared[j][threadIdx.x];
		}
		__syncthreads();
	}

	c[y * cWidth + x] = sum;
}

void matrixMulSTiled(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
	//checkCudaError(cudaFuncSetAttribute(matrixMulKernelTiled, cudaFuncAttributePreferredSharedMemoryCarveout, 50));
	const int tileSize = 16;
	assert(cWidth % tileSize == 0);
	assert(cHeight % tileSize == 0);
	assert(aWidth % tileSize == 0);
	assert(aWidth % 128 == 0); // row alligned with 512B block
	assert(cWidth % 128 == 0); // row alligned with 512B block
	dim3 blockSize(tileSize, tileSize);
	dim3 gridSize(cWidth / tileSize, cHeight / tileSize);
	matrixMulKernelSTiled<<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
}

__global__ void matrixMulKernelSTTiled(float* c, const float* a, const float* b, int cWidth, int aWidth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int tileSize = 4;
	const int blockSize = 16;
	assert(blockDim.x == blockSize && blockDim.y == blockSize);

	__shared__ float4 aShared[blockSize * tileSize * blockSize];
	__shared__ float4 bShared[blockSize * tileSize * blockSize];

	float4 row0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 row1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 row2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 row3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	for (int i = 0; i < aWidth; i += blockSize * tileSize) {
		#pragma unroll
		for (int j = 0; j < tileSize; j++) {
			aShared[(threadIdx.y * tileSize + j) * blockSize + threadIdx.x] = *((const float4*)(a + ((y * tileSize + j) * aWidth + (i + threadIdx.x * tileSize))));
			bShared[(threadIdx.y * tileSize + j) * blockSize + threadIdx.x] = *((const float4*)(b + ((i + threadIdx.y * tileSize + j) * cWidth + (x * tileSize))));
		}
		__syncthreads();

		float* aS = (float*)aShared + threadIdx.y * tileSize * blockSize * tileSize;
		float* bS = (float*)bShared + (threadIdx.x * tileSize);

        #pragma unroll
		for (int j = 0; j < blockSize * tileSize; j++) {
			float a0 = aS[0 * blockSize * tileSize + j];
			float a1 = aS[1 * blockSize * tileSize + j];
			float a2 = aS[2 * blockSize * tileSize + j];
			float a3 = aS[3 * blockSize * tileSize + j];
			float4 b = *((float4*)(bS + j * blockSize * tileSize));

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

	c = (c + y * tileSize * cWidth) + x * tileSize;
	*((float4*)c) = row0;
	c += cWidth;
	*((float4*)c) = row1;
	c += cWidth;
	*((float4*)c) = row2;
	c += cWidth;
	*((float4*)c) = row3;
}

void matrixMulSTTiled(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
	const int tileSize = 4;
	assert(cWidth % tileSize == 0);
	assert(cHeight % tileSize == 0);
	assert(aWidth % tileSize == 0);
	assert(aWidth % 128 == 0); // row alligned with 512B block
	assert(cWidth % 128 == 0); // row alligned with 512B block
	dim3 blockSize(16, 16);
	dim3 matPartSize(blockSize.x * tileSize, blockSize.y * tileSize);
	assert(cWidth % matPartSize.x == 0);
	assert(cHeight % matPartSize.y == 0);
	dim3 gridSize(cWidth / matPartSize.x, cHeight / matPartSize.y);
	matrixMulKernelSTTiled<<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
}

template <int width, int height, int threadCount>
__device__ void loadRect(float* dst, const float* src, int srcStride, int thread) {
	assert(threadCount % (width / 4) == 0);
	const int w = width / 4;
	const int h = threadCount / w;
	assert(threadCount % w == 0);

	int ty = thread / w;
	int tx = thread - (ty * w);
	tx *= 4;
	src += tx;
	dst += tx;

    #pragma unroll
	for (int i = ty; i < height; i += h) {
		reinterpret_cast<float4*>(dst)[i * w] = *reinterpret_cast<const float4*>(src + i * srcStride);
	}
}

__global__ void matrixMulKernelSWTiled(float* c, const float* a, const float* b, int cWidth, int aWidth)
{
	const int sharedSize = 32;
	const int tileWidth = 32;
	const int tileHeight = 64;
	const int gridWidth = 4;
	const int gridHeight = 2;
	assert(blockDim.x == tileWidth);
	assert(blockDim.y == gridWidth && blockDim.z == gridHeight);
	const int threadCount = tileWidth * gridWidth * gridHeight;

	int warpThread = threadIdx.x;
	int tileX = threadIdx.y;
	int tileY = threadIdx.z;
	int thread = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
	int x = blockIdx.x * blockDim.y * tileWidth;
	int y = blockIdx.y * blockDim.z * tileHeight;

	__shared__ float aS[gridHeight * tileHeight * sharedSize];
	__shared__ float bS[sharedSize * gridWidth * tileWidth];

	float accum[tileHeight];
	for (int i = 0; i < tileHeight; i++)
		accum[i] = 0;

	a += y * aWidth;
	b += x;

	for (int i = 0; i < aWidth; i += sharedSize) {
		loadRect<sharedSize, gridHeight * tileHeight, threadCount>((float*)aS, a + i, aWidth, thread);
		loadRect<gridWidth * tileWidth, sharedSize, threadCount>((float*)bS, b + i * cWidth, cWidth, thread);
		__syncthreads();

		float* a = aS + tileY * tileHeight * sharedSize;
		float* b = bS + tileX * tileWidth + warpThread;
        #pragma unroll
		for (int k = 0; k < sharedSize; k++) {
			float bValue = b[k * gridWidth * tileWidth];
			#pragma unroll
			for (int j = 0; j < tileHeight; j++) {
				accum[j] += a[j * sharedSize + k] * bValue;
			}
		}
		__syncthreads();
	}

	x += tileX * tileWidth;
	y += tileY * tileHeight;
	c += x + warpThread;
	c += y * cWidth;
	#pragma unroll
	for (int i = 0; i < tileHeight; i++)
		c[i * cWidth] = accum[i];
}

void matrixMulSWTiled(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
	const int sharedSize = 32;
	const int tileWidth = 32;
	const int tileHeight = 64;
	assert(tileWidth == 32); // tileWidth has to have warp size
	assert(aWidth % 128 == 0); // row alligned with 512B block
	assert(cWidth % 128 == 0); // row alligned with 512B block
	dim3 blockSize(32, 4, 2);
	dim3 accumSize(blockSize.y * tileWidth, blockSize.z * tileHeight);
	assert(aWidth % sharedSize == 0);
	assert(cWidth % accumSize.x == 0);
	assert(cHeight % accumSize.y == 0);
	dim3 gridSize(cWidth / accumSize.x, cHeight / accumSize.y);
	matrixMulKernelSWTiled<<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
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
		const int tileSize = 128;
		float accum[tileSize][tileSize];
		for (int ty = 0; ty < cHeight; ty += tileSize) {
			for (int tx = 0; tx < cWidth; tx += tileSize) {
				int tileWidth = std::min(cWidth - tx, tileSize);
				int tileHeight = std::min(cHeight - ty, tileSize);
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
	name += kernelName + "(";
	name += std::to_string(cWidth) + "x";
	name += std::to_string(cHeight);
	name += ", " + std::to_string(aWidth) + ")";
	printTestItem(name, time, pass);
}

void testMatrixMul() {
	checkCudaError(cudaSetDevice(0));

	testMatrixMul(1024, 1024, 1024, matrixMulNaive, "Naive");
	testMatrixMul(1024, 1024, 1024, matrixMulSTiled, "STiled");
	testMatrixMul(1024, 1024, 1024, matrixMulSTTiled, "STTiled");
	testMatrixMul(1024, 1024, 1024, matrixMulSWTiled, "SWTiled");

	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceReset());
}
