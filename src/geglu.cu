#include "geglu.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __inline__ float gelu(float x) {
	const float kA = 0.7978845608f; // sqrt(2 / PI)
	const float kC = 0.044715f;
	return 0.5f * x * (1.f + tanh(kA * (x + kC * x * x * x)));
}

__global__ void gegluKernel(float* out, const float* in, int batchSize, int dim)
{
	int outIdx = threadIdx.x + blockDim.x * blockIdx.x;
	int halfDim = dim / 2;
	if (outIdx < batchSize * halfDim) {
		int d = outIdx % halfDim;
		int b = outIdx / halfDim;
		int idx1 = b * dim + d;
		int idx2 = idx1 + halfDim;
		out[outIdx] = gelu(in[idx1]) * in[idx2];
	}
}

void geglu(float* out, const float* in, int batchSize, int dim) {
	const int threadsPerBlock = 256;
	int numberOfBlocks = ((batchSize * dim / 2) + threadsPerBlock - 1) / threadsPerBlock;
	gegluKernel<<<numberOfBlocks, threadsPerBlock>>>(out, in, batchSize, dim);
}

using KernelFunction = void(*)(float*, const float*, int, int);

float geglu(CudaBuffer<float>& outTensor, const CudaBuffer<float>& inTensor,
	int batchSize, int dim, KernelFunction kernelFunc) {
	CudaBuffer<float> outTensorDev(outTensor.size(), cudaMemoryTypeDevice);
	CudaBuffer<float> inTensorDev(inTensor.size(), cudaMemoryTypeDevice);
	inTensorDev.copyFrom(inTensor);

	float elapsedTime;
	CudaEvent start, stop;
	checkCudaError(cudaEventRecord(start(), 0));
	kernelFunc(outTensorDev(), inTensorDev(), batchSize, dim);
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaEventRecord(stop(), 0));
	checkCudaError(cudaEventSynchronize(stop()));
	checkCudaError(cudaEventElapsedTime(&elapsedTime, start(), stop()));

	checkCudaError(cudaDeviceSynchronize());
	outTensor.copyFrom(outTensorDev);
	return elapsedTime;
}

float geluRef(float x) {
	const float kPI = 3.14159265359f;
	return 0.5f * x * (1.f + tanh(sqrt(2.f / kPI) * (x + 0.044715f * pow(x, 3.f))));
}

void gegluRef(float* out, const float* in, int batchSize, int dim) {
	for (int b = 0; b < batchSize; b++) {
		for (int d = 0; d < dim / 2; d++) {
			int idx1 = b * dim + d;
			int idx2 = idx1 + dim / 2;
			out[b * dim / 2 + d] = geluRef(in[idx1]) * in[idx2];
		}
	}
}

void testGeGLU(int batchSize, int dim, KernelFunction kernelFunc) {
	assert(batchSize > 0);
	assert(dim > 0 && dim % 2 == 0);

	CudaBuffer<float> inTensor(batchSize * dim, cudaMemoryTypeHost);
	inTensor.fillWithRandom();

	const int outTensorSize = inTensor.size() / 2;
	CudaBuffer<float> outTensor(outTensorSize, cudaMemoryTypeHost);
	float time = geglu(outTensor, inTensor, batchSize, dim, kernelFunc);
	CudaBuffer<float> outTensorRef(outTensorSize, cudaMemoryTypeHost);
	gegluRef(outTensorRef(), inTensor(), batchSize, dim);

	bool pass = outTensor.approxEqual(outTensorRef);
	std::string name = "GeGLU(";
	name += std::to_string(batchSize) + "x";
	name += std::to_string(dim) += ")";
	printTestItem(name, time, pass);
}

void testGeGLU() {
	checkCudaError(cudaSetDevice(0));

	testGeGLU(64, 4096, geglu);
	testGeGLU(256, 4096, geglu);

	checkCudaError(cudaDeviceReset());
}
