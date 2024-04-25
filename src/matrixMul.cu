#include "matrixMul.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matrixMulKernelNaive(float* c, float* a, float* b, int cWidth, int cHeight, int aWidth)
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

void matrixMulNaive(float* c, float* a, float* b, int cWidth, int cHeight, int aWidth) {
	dim3 blockSize(32, 32);
	dim3 gridSize((cWidth + blockSize.x - 1) / blockSize.x,
		(cHeight + blockSize.y - 1) / blockSize.y);
	matrixMulKernelNaive<<<gridSize, blockSize>>>(c, a, b, cWidth, cHeight, aWidth);
}

__global__ void matrixMulKernelTiled(float* c, float* a, float* b, int cWidth, int aWidth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int tileSize = 32;
	assert(blockDim.x == tileSize && blockDim.y == tileSize);

	const int sharedSize = tileSize * tileSize;
	__shared__ float aShared[sharedSize];
	__shared__ float bShared[sharedSize];

	float sum = 0;

	for (int i = 0; i < aWidth; i += tileSize) {
		int sharedIdx = threadIdx.y * tileSize + threadIdx.x;
		aShared[sharedIdx] = a[y * aWidth + i + threadIdx.x];
		bShared[sharedIdx] = b[(i + threadIdx.y) * cWidth + x];
		__syncthreads();

		for (int j = 0; j < tileSize; j++) {
			sum += aShared[threadIdx.y * tileSize + j] * bShared[j * tileSize + threadIdx.x];
		}
		__syncthreads();
	}

	c[y * cWidth + x] = sum;
}

void matrixMulTiled(float* c, float* a, float* b, int cWidth, int cHeight, int aWidth) {
	const int tileSize = 32;
	assert(cWidth % tileSize == 0);
	assert(cHeight % tileSize == 0);
	assert(aWidth % tileSize == 0);
	dim3 blockSize(tileSize, tileSize);
	dim3 gridSize(cWidth / tileSize, cHeight / tileSize);
	matrixMulKernelTiled<<<gridSize, blockSize>>>(c, a, b, cWidth, aWidth);
}

using KernelFunction = void(*)(float*, float*, float*, int, int, int);

float matrixMul(CudaBuffer<float>& cHost, CudaBuffer<float>& aHost, CudaBuffer<float>& bHost,
	            int cWidth, int cHeight, int aWidth, KernelFunction kernelFunc) {
	CudaBuffer<float> aDev(aWidth * cHeight, cudaMemoryTypeDevice);
	CudaBuffer<float> bDev(cWidth * aWidth, cudaMemoryTypeDevice);
	CudaBuffer<float> cDev(cWidth * cHeight, cudaMemoryTypeDevice);
	aDev.copyFrom(aHost);
	bDev.copyFrom(bHost);

	float elapsedTime;
	CudaEvent start, stop;
	checkCudaError(cudaEventRecord(start(), 0));
	kernelFunc(cDev(), aDev(), bDev(), cWidth, cHeight, aWidth);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaEventRecord(stop(), 0));
	checkCudaError(cudaEventSynchronize(stop()));
	checkCudaError(cudaEventElapsedTime(&elapsedTime, start(), stop()));

	checkCudaError(cudaDeviceSynchronize());
	cHost.copyFrom(cDev);

	return elapsedTime;
}

void matrixMulRef(float* c, float* a, float* b, int cWidth, int cHeight, int aWidth) { // row-major order
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

void testMatrixMul(int cWidth, int cHeight, int aWidth, KernelFunction kernelFunc) {
	CudaBuffer<float> a(aWidth * cHeight, cudaMemoryTypeHost);
	CudaBuffer<float> b(cWidth * aWidth, cudaMemoryTypeHost);
	a.fillWithRandom();
	b.fillWithRandom();

	CudaBuffer<float> c(cWidth * cHeight, cudaMemoryTypeHost);
	float time = matrixMul(c, a, b, cWidth, cHeight, aWidth, kernelFunc);
	CudaBuffer<float> cRef(cWidth * cHeight, cudaMemoryTypeHost);
	matrixMulRef(cRef(), a(), b(), cWidth, cHeight, aWidth);

	bool pass = c.approxEqual(cRef);
	std::string name = "MatrixMul(";
	name += std::to_string(cWidth) + "x";
	name += std::to_string(cHeight);
	name += ", " + std::to_string(aWidth) + ")";
	printTestItem(name, time, pass);
}

void testMatrixMul() {
	checkCudaError(cudaSetDevice(0));

	testMatrixMul(1024, 1024, 1024, matrixMulNaive);
	testMatrixMul(1024, 1024, 1024, matrixMulTiled);

	checkCudaError(cudaDeviceReset());
}
