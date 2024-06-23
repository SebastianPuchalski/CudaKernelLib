#include "geglu.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <iomanip>

__device__ __inline__ float gelu(float x) {
	const float A = 0.7978845608f; // sqrt(2 / PI)
	const float C = 0.044715f;
	return 0.5f * x * (1.f + tanh(A * (x + C * x * x * x)));
}

__global__ void gegluKernel(float* out, const float* in, int outSize, int halfDim)
{
	int outIdx = threadIdx.x + blockDim.x * blockIdx.x;
	if (outIdx < outSize) {
		int idx1 = outIdx * 2 - outIdx % halfDim;
		int idx2 = idx1 + halfDim;
		out[outIdx] = gelu(in[idx1]) * in[idx2];
	}
}

void geglu(float* out, const float* in, int batchSize, int dim) {
	assert(dim % 2 == 0);
	int halfDim = dim / 2;
	int outSize = batchSize * halfDim;
	const int BLOCK_SIZE = 256;
	int gridSize = (outSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
	gegluKernel<<<gridSize, BLOCK_SIZE>>>(out, in, outSize, halfDim);
}

float geglu(CudaBuffer<float>& outTensor,
			const CudaBuffer<float>& inTensor,
			int batchSize, int dim) {
	CudaBuffer<float> outTensorDev(outTensor.size(), cudaMemoryTypeDevice);
	CudaBuffer<float> inTensorDev(inTensor.size(), cudaMemoryTypeDevice);
	inTensorDev.copyFrom(inTensor);

	CudaEvent start, stop;
	start.record();
	geglu(outTensorDev(), inTensorDev(), batchSize, dim);
	checkCudaError(cudaGetLastError());
	stop.record();
	stop.synchronize();
	float elapsedTime = start.elapsedTime(stop);

	checkCudaError(cudaDeviceSynchronize());
	outTensor.copyFrom(outTensorDev);
	return elapsedTime;
}

float geluRef(float x) {
	const float PI = 3.14159265359f;
	return 0.5f * x * (1.f + tanh(sqrt(2.f / PI) * (x + 0.044715f * pow(x, 3.f))));
}

void gegluRef(float* out, const float* in, int batchSize, int dim) {
	assert(dim % 2 == 0);
	for (int b = 0; b < batchSize; b++) {
		for (int d = 0; d < dim / 2; d++) {
			int idx1 = b * dim + d;
			int idx2 = idx1 + dim / 2;
			out[b * dim / 2 + d] = geluRef(in[idx1]) * in[idx2];
		}
	}
}

void testGeGLU(int batchSize, int dim) {
	assert(batchSize > 0);
	assert(dim > 0 && dim % 2 == 0);

	CudaBuffer<float> inTensor(batchSize * dim, cudaMemoryTypeHost);
	inTensor.fillWithRandom();

	const int outTensorSize = inTensor.size() / 2;
	CudaBuffer<float> outTensor(outTensorSize, cudaMemoryTypeHost);
	float time = geglu(outTensor, inTensor, batchSize, dim);
	CudaBuffer<float> outTensorRef(outTensorSize, cudaMemoryTypeHost);
	gegluRef(outTensorRef(), inTensor(), batchSize, dim);

	bool pass = outTensor.approxEqual(outTensorRef);
	std::string name = "GeGLU (";
	name += std::to_string(batchSize) + ", ";
	name += std::to_string(dim) += ")";

	float theoreticalTime = getBestMemoryAccessTime(outTensorSize * sizeof(float));
	theoreticalTime *= 3 * 1000; // read input and write output in milliseconds
	float efficiency = theoreticalTime / time;
	std::ostringstream addInfo;
	addInfo << std::fixed << std::setprecision(0);
	addInfo << efficiency * 100 << "%";

	printTestItem(name, pass, time, addInfo.str());
	checkCudaError(cudaDeviceReset());
}

void testGeGLU() {
	checkCudaError(cudaSetDevice(0));

	testGeGLU(64, 1024 * 128);
	testGeGLU(27, 999 * 236);
	testGeGLU(256, 1024 * 128);
	testGeGLU(1024, 1024 * 128);
}
