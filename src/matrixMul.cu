#include "matrixMul.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matrixMulKernel(float* c, float* a, float* b, int cWidth, int cHeight, int aWidth)
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

float matrixMul(CudaBuffer<float>& cHost, CudaBuffer<float>& aHost, CudaBuffer<float>& bHost,
	            int cWidth, int cHeight, int aWidth) {
	CudaBuffer<float> aDev(aWidth * cHeight, cudaMemoryTypeDevice);
	CudaBuffer<float> bDev(cWidth * aWidth, cudaMemoryTypeDevice);
	CudaBuffer<float> cDev(cWidth * cHeight, cudaMemoryTypeDevice);
	aDev.copyFrom(aHost);
	bDev.copyFrom(bHost);

	dim3 blockSize(16, 16);
	dim3 gridSize((cWidth + blockSize.x - 1) / blockSize.x,
		          (cHeight + blockSize.y - 1) / blockSize.y);

	float elapsedTime;
	CudaEvent start, stop;
	checkCudaError(cudaEventRecord(start(), 0));
	matrixMulKernel<<<gridSize, blockSize>>> (cDev(), aDev(), bDev(), cWidth, cHeight, aWidth);
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

void testMatrixMul(int cWidth, int cHeight, int aWidth) {
	CudaBuffer<float> a(aWidth * cHeight, cudaMemoryTypeHost);
	CudaBuffer<float> b(cWidth * aWidth, cudaMemoryTypeHost);
	a.fillWithRandom();
	b.fillWithRandom();

	CudaBuffer<float> c(cWidth * cHeight, cudaMemoryTypeHost);
	CudaBuffer<float> cRef(cWidth * cHeight, cudaMemoryTypeHost);
	float time = matrixMul(c, a, b, cWidth, cHeight, aWidth);
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
	testMatrixMul(512, 256, 1024);
	checkCudaError(cudaDeviceReset());
}
