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

__global__ void matrixMulKernelTiled(float* c, const float* a, const float* b, int cWidth, int aWidth)
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

void matrixMulTiled(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
	//checkCudaError(cudaFuncSetAttribute(matrixMulKernelTiled, cudaFuncAttributePreferredSharedMemoryCarveout, 50));
	const int tileSize = 16;
	assert(cWidth % tileSize == 0);
	assert(cHeight % tileSize == 0);
	assert(aWidth % tileSize == 0);
	assert(aWidth % 128 == 0); // row alligned with 512B block
	assert(cWidth % 128 == 0); // row alligned with 512B block
	dim3 blockSize(tileSize, tileSize);
	dim3 gridSize(cWidth / tileSize, cHeight / tileSize);
	matrixMulKernelTiled<<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
}

__global__ void matrixMulKernelTiledT(float* c, const float* a, const float* b, int cWidth, int aWidth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

	const int tileSize = 32;
	assert(blockDim.x == tileSize && blockDim.y * 2 == tileSize);

	__shared__ float aShared[tileSize * tileSize * 2];
	__shared__ float bShared[tileSize * tileSize];

	float4 sum = make_float4(0, 0, 0, 0);

	for (int i = 0; i < aWidth; i += tileSize) {
        #pragma unroll
		for (int k = 0; k < 4; k++)
			aShared[threadIdx.x * tileSize * 2 + (threadIdx.y * 4 + k)] = a[(y + k) * aWidth + (i + threadIdx.x)];
		#pragma unroll
		for (int k = 0; k < 2; k++)
			bShared[(threadIdx.y * 2 + k) * tileSize + threadIdx.x] = b[(i + (threadIdx.y * 2 + k)) * cWidth + x];
		__syncthreads();

		#pragma unroll
		for (int j = 0; j < tileSize; j++) {
			float bVal = bShared[j * tileSize + threadIdx.x];
			int index = j * tileSize / 2 + threadIdx.y;
			float4 aVal = ((float4*)aShared)[index];
			sum.x += aVal.x * bVal;
			sum.y += aVal.y * bVal;
			sum.z += aVal.z * bVal;
			sum.w += aVal.w * bVal;
		}
		__syncthreads();
	}

	c[(y + 0) * cWidth + x] = sum.x;
	c[(y + 1) * cWidth + x] = sum.y;
	c[(y + 2) * cWidth + x] = sum.z;
	c[(y + 3) * cWidth + x] = sum.w;
}

void matrixMulTiledT(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
	//checkCudaError(cudaFuncSetAttribute(matrixMulKernelTiled, cudaFuncAttributePreferredSharedMemoryCarveout, 50));
	const int tileSize = 32;
	assert(cWidth % tileSize == 0);
	assert(cHeight % tileSize == 0);
	assert(aWidth % tileSize == 0);
	assert(aWidth % 128 == 0); // row alligned with 512B block
	assert(cWidth % 128 == 0); // row alligned with 512B block
	dim3 blockSize(tileSize, tileSize / 2);
	dim3 gridSize(cWidth / tileSize, cHeight / tileSize / 2);
	matrixMulKernelTiledT<<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
}

__global__ void matrixMulKernelFast(float* c, const float* a, const float* b, int cWidth, int aWidth)
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

		float* a = (float*)aShared;
		float* b = (float*)bShared;

        #pragma unroll
		for (int j = 0; j < blockSize * tileSize; j++) {
			row0.x += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row0.y += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row0.z += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row0.w += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 3)];

			row1.x += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row1.y += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row1.z += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row1.w += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 3)];

			row2.x += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row2.y += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row2.z += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row2.w += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 3)];

			row3.x += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row3.y += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row3.z += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row3.w += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 3)];
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

void matrixMulFast(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
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
	matrixMulKernelFast<<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
}

__global__ void matrixMulKernelFastTest(float* c, const float* a, const float* b, int cWidth, int aWidth)
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
			/*aShared[(threadIdx.y * tileSize + j) * blockSize + threadIdx.x] = make_float4(y, j, i, threadIdx.x);
			bShared[(threadIdx.y * tileSize + j) * blockSize + threadIdx.x] = make_float4(i, threadIdx.y, j, x);*/
		}
		__syncthreads();

		float* a = (float*)aShared;
		float* b = (float*)bShared;

#		//pragma unroll
		//for (int j = 0; j < blockSize * tileSize; j++) {
			/*row0.x += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row0.y += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row0.z += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row0.w += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 3)];

			row1.x += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row1.y += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row1.z += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row1.w += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 3)];

			row2.x += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row2.y += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row2.z += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row2.w += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 3)];

			row3.x += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row3.y += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row3.z += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row3.w += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + j] * b[j * blockSize * tileSize + (threadIdx.x * tileSize + 3)];*/

			row0.x += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + (threadIdx.x * tileSize + 0)] * b[(threadIdx.y * tileSize + 0) * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row0.y += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + (threadIdx.x * tileSize + 1)] * b[(threadIdx.y * tileSize + 0) * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row0.z += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + (threadIdx.x * tileSize + 2)] * b[(threadIdx.y * tileSize + 0) * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row0.w += a[(threadIdx.y * tileSize + 0) * blockSize * tileSize + (threadIdx.x * tileSize + 3)] * b[(threadIdx.y * tileSize + 0) * blockSize * tileSize + (threadIdx.x * tileSize + 3)];

			row1.x += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + (threadIdx.x * tileSize + 0)] * b[(threadIdx.y * tileSize + 1) * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row1.y += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + (threadIdx.x * tileSize + 1)] * b[(threadIdx.y * tileSize + 1) * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row1.z += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + (threadIdx.x * tileSize + 2)] * b[(threadIdx.y * tileSize + 1) * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row1.w += a[(threadIdx.y * tileSize + 1) * blockSize * tileSize + (threadIdx.x * tileSize + 3)] * b[(threadIdx.y * tileSize + 1) * blockSize * tileSize + (threadIdx.x * tileSize + 3)];

			row2.x += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + (threadIdx.x * tileSize + 0)] * b[(threadIdx.y * tileSize + 2) * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row2.y += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + (threadIdx.x * tileSize + 1)] * b[(threadIdx.y * tileSize + 2) * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row2.z += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + (threadIdx.x * tileSize + 2)] * b[(threadIdx.y * tileSize + 2) * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row2.w += a[(threadIdx.y * tileSize + 2) * blockSize * tileSize + (threadIdx.x * tileSize + 3)] * b[(threadIdx.y * tileSize + 2) * blockSize * tileSize + (threadIdx.x * tileSize + 3)];

			row3.x += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + (threadIdx.x * tileSize + 0)] * b[(threadIdx.y * tileSize + 3) * blockSize * tileSize + (threadIdx.x * tileSize + 0)];
			row3.y += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + (threadIdx.x * tileSize + 1)] * b[(threadIdx.y * tileSize + 3) * blockSize * tileSize + (threadIdx.x * tileSize + 1)];
			row3.z += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + (threadIdx.x * tileSize + 2)] * b[(threadIdx.y * tileSize + 3) * blockSize * tileSize + (threadIdx.x * tileSize + 2)];
			row3.w += a[(threadIdx.y * tileSize + 3) * blockSize * tileSize + (threadIdx.x * tileSize + 3)] * b[(threadIdx.y * tileSize + 3) * blockSize * tileSize + (threadIdx.x * tileSize + 3)];
		//}
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

void matrixMulFastTest(float* c, const float* a, const float* b, int cWidth, int cHeight, int aWidth) {
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
	matrixMulKernelFastTest<<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
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
	              int cWidth, int cHeight, int aWidth) { // row-major order
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
	//matrixMulRef(cRef(), a(), b(), cWidth, cHeight, aWidth);

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

	int m = 1;
	testMatrixMul(1024 * m, 1024 * m, 1024 * m, matrixMulNaive, "Naive");
	testMatrixMul(1024 * m, 1024 * m, 1024 * m, matrixMulTiled, "Tiled");
	testMatrixMul(1024 * m, 1024 * m, 1024 * m, matrixMulTiledT, "TiledT");
	testMatrixMul(1024*m, 1024*m, 1024*m, matrixMulFast, "Fast");
	testMatrixMul(1024*m, 1024*m, 1024*m, matrixMulFastTest, "FastTest");

	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceReset());
}
