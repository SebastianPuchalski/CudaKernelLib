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

float matrixMul(float* c, float* a, float* b, int cWidth, int cHeight, int aWidth) {
	CudaBuffer<float> aBuff(aWidth * cHeight);
	CudaBuffer<float> bBuff(cWidth * aWidth);
	CudaBuffer<float> cBuff(cWidth * cHeight);

	checkCudaError(cudaMemcpy(aBuff(), a, aBuff.dataSize(), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(bBuff(), b, bBuff.dataSize(), cudaMemcpyHostToDevice));

	dim3 blockSize(16, 16);
	dim3 gridSize((cWidth + blockSize.x - 1) / blockSize.x,
		          (cHeight + blockSize.y - 1) / blockSize.y);

	float elapsedTime;
	CudaEvent start, stop;
	checkCudaError(cudaEventRecord(start(), 0));
	matrixMulKernel<<<gridSize, blockSize>>> (cBuff(), aBuff(), bBuff(), cWidth, cHeight, aWidth);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaEventRecord(stop(), 0));
	checkCudaError(cudaEventSynchronize(stop()));
	checkCudaError(cudaEventElapsedTime(&elapsedTime, start(), stop()));

	checkCudaError(cudaDeviceSynchronize());
	checkCudaError(cudaMemcpy(c, cBuff(), cBuff.dataSize(), cudaMemcpyDeviceToHost));

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
	std::vector<float> a(aWidth * cHeight);
	std::vector<float> b(cWidth * aWidth);
	fillVectorRandom(a);
	fillVectorRandom(b);

	std::vector<float> c(cWidth * cHeight);
	std::vector<float> cRef(cWidth * cHeight);
	float time = matrixMul(c.data(), a.data(), b.data(), cWidth, cHeight, aWidth);
	matrixMulRef(cRef.data(), a.data(), b.data(), cWidth, cHeight, aWidth);

	bool pass = compareVectors(c, cRef);
	std::string name = "MatrixMul(";
	name += std::to_string(cWidth) + "x";
	name += std::to_string(cHeight);
	name += ", " + std::to_string(aWidth) + ")";
	printTestItem(name, time, pass);
}

void testMatrixMul() {
	checkCudaError(cudaSetDevice(0));
	testMatrixMul(512, 256, 1024);
	checkCudaError(cudaDeviceReset()); // is this a right place?
}
