#include "softmax.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <iomanip>

const int BASE_DIM_NUMBER = 2;
const int EXT_DIM_NUMBER = 4;
const int MAX_DIM_NUMBER = 16; // it can be any number
__constant__ int addDimSizes[MAX_DIM_NUMBER - EXT_DIM_NUMBER];

__inline__ __device__ int64_t calcAddOffset(int idxEven, int idxOdd, int dimNumber) {
	int64_t offset = 0;
	int stride = 1;
	bool even = true;
	for (int i = 0; i < dimNumber; i++) {
		int& idx = even ? idxEven : idxOdd;
		even = !even;
		int size = addDimSizes[i];
		int div = idx / size;
		int mod = idx - div * size;
		idx = div;
		offset += mod * stride;
		stride *= size;
	}
	return offset;
}

template <int MAX_DIM_NUM = MAX_DIM_NUMBER>
__global__ void softmaxKernelM(float* out, const float* in, int size0, int size2, int sizeEven, int addDimNumber) {
	const int WARP_SIZE = 32;
	assert(WARP_SIZE == warpSize);
	const int thread = threadIdx.x;

	const int size1 = gridDim.x;
	const int size3 = gridDim.y;
	const int sizeOdd = gridDim.x;
	const int idx1 = blockIdx.x;
	const int idx3 = blockIdx.y;
	const int idxOdd = blockIdx.z;

	const int stride0 = size0;
	const int stride01 = stride0 * size1;
	const int stride02 = stride0 * size2;
	const int stride012 = stride01 * size2;
	const int stride0123 = stride012 * size3;

	if (MAX_DIM_NUM <= EXT_DIM_NUMBER) {
		sizeEven = 1;
		addDimNumber = 0;
	}
	if (MAX_DIM_NUM <= BASE_DIM_NUMBER) {
		size2 = 1;
	}

	extern __shared__ float shared[];

	const int64_t offset13 = idx1 * stride0 + idx3 * stride012;
	in += offset13;
	out += offset13;

	float max = FLT_MIN;
	for (int idxEven = 0; idxEven < sizeEven; idxEven++) {
		float* lShared = shared;
		const float* lIn = in;
		if (addDimNumber) {
			lShared += idxEven * stride02;
			lIn += calcAddOffset(idxEven, idxOdd, addDimNumber) * stride0123;
		}
		for (int idx2 = 0; idx2 < size2; idx2++) {
			for (int idx0 = thread; idx0 < size0; idx0 += WARP_SIZE) {
				const float value = lIn[idx0];
				lShared[idx0] = value;
				max = fmaxf(value, max);
			}
			lIn += stride01;
			lShared += stride0;
		}
	}
	#pragma unroll
	for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
		max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, i));

	float sum = 0;
	for (int idxEven = 0; idxEven < sizeEven; idxEven++) {
		float* lShared = shared + idxEven * stride02;
		for (int idx2 = 0; idx2 < size2; idx2++) {
			for (int idx0 = thread; idx0 < size0; idx0 += WARP_SIZE) {
				const float value = expf(lShared[idx0] - max);
				lShared[idx0] = value;
				sum += value;
			}
			lShared += stride0;
		}
	}
	#pragma unroll
	for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
		sum += __shfl_xor_sync(0xffffffff, sum, i);

	const float invSum = 1.f / sum;
	for (int idxEven = 0; idxEven < sizeEven; idxEven++) {
		float* lShared = shared;
		float* lOut = out;
		if (addDimNumber) {
			lShared += idxEven * stride02;
			lOut += calcAddOffset(idxEven, idxOdd, addDimNumber) * stride0123;
		}
		for (int idx2 = 0; idx2 < size2; idx2++) {
			for (int idx0 = thread; idx0 < size0; idx0 += WARP_SIZE)
				lOut[idx0] = lShared[idx0] * invSum;
			lOut += stride01;
			lShared += stride0;
		}
	}
}

__global__ void softmaxKernelM(float* out, const float* in, int size0) {
	const int WARP_SIZE = 32;
	assert(WARP_SIZE == warpSize);
	const int thread = threadIdx.x;

	extern __shared__ float shared[];

	const int64_t offset1 = blockIdx.x * size0;
	in += offset1;
	out += offset1;

	float max = FLT_MIN;
	for (int i = thread; i < size0; i += WARP_SIZE) {
		const float value = in[i];
		shared[i] = value;
		max = fmaxf(value, max);
	}
	#pragma unroll
	for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
		max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, i));

	float sum = 0;
	for (int i = thread; i < size0; i += WARP_SIZE) {
		const float value = expf(shared[i] - max);
		shared[i] = value;
		sum += value;
	}
	#pragma unroll
	for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
		sum += __shfl_xor_sync(0xffffffff, sum, i);

	const float invSum = 1.f / sum;
	for (int i = thread; i < size0; i += WARP_SIZE)
		out[i] = shared[i] * invSum;
}

template <int BLOCK_SIZE>
__inline__ __device__ void reduceMaxSumInBlock(float& max, float& sum, int thread) {
	assert(BLOCK_SIZE % 2 == 0);
	const int WARP_SIZE = 32;

	__shared__ float maxMem[BLOCK_SIZE];
	__shared__ float sumMem[BLOCK_SIZE];
	maxMem[thread] = max;
	sumMem[thread] = sum;
	__syncthreads();

	#pragma unroll
	for (int i = BLOCK_SIZE / 2; i >= WARP_SIZE; i >>= 1) {
		assert(i % 2 == 0);
		if (thread < i) {
			const float max1 = maxMem[thread];
			const float max2 = maxMem[thread + i];
			float sum1 = sumMem[thread];
			float sum2 = sumMem[thread + i];
			const float newMax = fmaxf(max1, max2);
			sum1 *= __expf(max1 - newMax);
			sum2 *= __expf(max2 - newMax);
			maxMem[thread] = newMax;
			sumMem[thread] = sum1 + sum2;
		}
		__syncthreads();
	}

	if (thread < WARP_SIZE) {
		float max = maxMem[thread];
		float sum = sumMem[thread];

		float newMax = max;
		#pragma unroll
		for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
			newMax = fmaxf(newMax, __shfl_xor_sync(0xffffffff, newMax, i));
		maxMem[thread] = newMax;

		sum *= __expf(max - newMax);
		for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
			sum += __shfl_xor_sync(0xffffffff, sum, i);
		sumMem[thread] = sum;
	}
	__syncthreads();
	max = *maxMem;
	sum = *sumMem;
}

template <int BLOCK_SIZE>
__global__ void softmaxKernelL(float* out, const float* in, int size0) {
	const int WARP_SIZE = 32;
	assert(WARP_SIZE == warpSize);
	const int thread = threadIdx.x;

	const int64_t offset1 = blockIdx.x * size0;
	in += offset1;
	out += offset1;

	float max = in[thread];
	float sum = 1;
	for (int i = thread + BLOCK_SIZE; i < size0; i += BLOCK_SIZE) {
		const float value = in[i];
		const float newMax = fmaxf(max, value);
		const float fraction = __expf(fminf(max, value) - newMax);
		sum = (value > max) ? sum * fraction + 1 : sum + fraction;
		/*sum *= expf(max - newMax);
		sum += expf(value - newMax);*/
		max = newMax;
	}

	reduceMaxSumInBlock<BLOCK_SIZE>(max, sum, thread);

	const float invSum = 1.f / sum;
	for (int i = thread; i < size0; i += BLOCK_SIZE)
		out[i] = __expf(in[i] - max) * invSum;
}

bool softmaxL(float* out, const float* in, std::vector<int> dimensions) {
	const int WARP_SIZE = 32;
	const int BLOCK_SIZE = 1024;

	assert(!dimensions.empty());
	int size0 = dimensions[0];
	int size1 = dimensions.size() > 1 ? dimensions[1] : 1;
	if (dimensions.size() <= BASE_DIM_NUMBER) {
		dim3 blockSize(BLOCK_SIZE);
		dim3 gridSize(size1);
		softmaxKernelL<BLOCK_SIZE><<<gridSize, blockSize>>>(out, in, size0);
		return true;
	}
}

void softmaxM(float* out, const float* in, std::vector<int> dimensions) {
	const int WARP_SIZE = 32;

	assert(!dimensions.empty());
	int size0 = dimensions[0];
	assert(size0 > 1);
	int size1 = dimensions.size() > 1 ? dimensions[1] : 1;
	/*if (dimensions.size() <= BASE_DIM_NUMBER) {
		dim3 blockSize(WARP_SIZE);
		dim3 gridSize(size1);
		int sharedSize = size0 * sizeof(float);
		assert(sharedSize < 48 * 1024);
		softmaxKernel<<<gridSize, blockSize, sharedSize>>>(out, in, size0);
	}*/
	int size2 = dimensions.size() > 2 ? dimensions[2] : 1;
	int size3 = dimensions.size() > 3 ? dimensions[3] : 1;
	int addDimNum = 0;
	int sizeEven = 1;
	int sizeOdd = 1;
	if (dimensions.size() > EXT_DIM_NUMBER) {
		dimensions.erase(dimensions.begin(), dimensions.begin() + EXT_DIM_NUMBER);
		addDimNum = dimensions.size();
		assert(addDimNum + EXT_DIM_NUMBER <= MAX_DIM_NUMBER);
		cudaMemcpyToSymbol(addDimSizes, dimensions.data(), dimensions.size() * sizeof(int));
		for (int i = 0; i < dimensions.size(); i++) {
			assert(dimensions[i] > 1);
			if (i % 2 == 0)
				sizeEven *= dimensions[i];
			else
				sizeOdd *= dimensions[i];
		}
	}
	dim3 blockSize(WARP_SIZE);
	assert(sizeOdd < 65536);
	dim3 gridSize(size1, size3, sizeOdd);
	int sharedSize = size0 * size2 * sizeEven * sizeof(float);
	assert(sharedSize < 48 * 1024);
	if(addDimNum) {
		softmaxKernelM<<<gridSize, blockSize, sharedSize>>>(out, in, size0, size2, sizeEven, addDimNum);
	}
	else {
		if (size2 > 1 || size3 > 1) {
			softmaxKernelM<EXT_DIM_NUMBER><<<gridSize, blockSize, sharedSize>>>(out, in, size0, size2, sizeEven, addDimNum);
		}
		else {
			softmaxKernelM<BASE_DIM_NUMBER><<<gridSize, blockSize, sharedSize>>>(out, in, size0, size2, sizeEven, addDimNum);
		}
	}
}

//-----------------------------------------------------------------------------

//const int WIDTH = 1024;
//
//__inline__ __device__ float reduceMax(float value, int thread) {
//	const int WARP_SIZE = 32;
//	__shared__ float reduceMem[WIDTH];
//	reduceMem[thread] = value;
//	__syncthreads();
//
//#pragma unroll
//	for (int i = WIDTH / 2; i >= WARP_SIZE; i >>= 1) {
//		if (thread < i)
//			reduceMem[thread] = fmaxf(reduceMem[thread], reduceMem[thread + i]);
//		__syncthreads();
//	}
//
//	if (thread < WARP_SIZE) {
//		float max = reduceMem[thread];
//#pragma unroll
//		for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
//			max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, i));
//		reduceMem[thread] = max;
//	}
//	__syncthreads();
//
//	return reduceMem[0];
//}
//
//__inline__ __device__ float reduceSum(float value, int thread) {
//	const int WARP_SIZE = 32;
//	__shared__ float reduceMem[WIDTH];
//	reduceMem[thread] = value;
//	__syncthreads();
//
//#pragma unroll
//	for (int i = WIDTH / 2; i >= WARP_SIZE; i >>= 1) {
//		if (thread < i)
//			reduceMem[thread] = reduceMem[thread] + reduceMem[thread + i];
//		__syncthreads();
//	}
//
//	if (thread < WARP_SIZE) {
//		float sum = reduceMem[thread];
//#pragma unroll
//		for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
//			sum += __shfl_xor_sync(0xffffffff, sum, i);
//		reduceMem[thread] = sum;
//	}
//	__syncthreads();
//
//	return reduceMem[0];
//}
//
//__global__ void softmaxTestKernel(float* out, const float* in, int size0) {
//	const int WARP_SIZE = 32;
//	assert(WARP_SIZE == warpSize);
//	const int thread = threadIdx.x;
//
//	const int64_t offset1 = blockIdx.x * size0;
//	in += offset1;
//	out += offset1;
//
//	float max = FLT_MIN;
//	for (int i = thread; i < size0; i += WIDTH)
//		max = fmaxf(in[i], max);
//	max = reduceMax(max, thread);
//
//	float sum = 0;
//	for (int i = thread; i < size0; i += WIDTH)
//		sum += expf(in[i] - max);
//	sum = reduceSum(sum, thread);
//
//	sum = 1.f / sum;
//	for (int i = thread; i < size0; i += WIDTH)
//		out[i] = expf(in[i] - max) * sum;
//}
//
//bool softmaxTest(float* out, const float* in, std::vector<int> dimensions) {
//	const int WARP_SIZE = 32;
//
//	assert(!dimensions.empty());
//	int size0 = dimensions[0];
//	int size1 = dimensions.size() > 1 ? dimensions[1] : 1;
//	if (dimensions.size() <= BASE_DIM_NUMBER) {
//		dim3 blockSize(WIDTH);
//		dim3 gridSize(size1);
//		softmaxTestKernel << <gridSize, blockSize >> > (out, in, size0);
//		return true;
//	}
//} // 256: 53%. 512: 47%

//-----------------------------------------------------------------------------

void flattenTensor(std::vector<int>& dimensions, std::vector<bool>& mask) {
	assert(dimensions.size() == mask.size() && !dimensions.empty());
	std::vector<int> newDimensions;
	std::vector<bool> newMask;
	bool prev = !mask.front();
	for (int i = 0; i < dimensions.size(); i++) {
		assert(dimensions[i] > 0);
		if (mask[i] == prev) {
			newDimensions.back() *= dimensions[i];
		}
		else {
			newDimensions.push_back(dimensions[i]);
			newMask.push_back(mask[i]);
		}
		prev = mask[i];
	}
	dimensions = newDimensions;
	mask = newMask;
}

void softmax(float* out, const float* in,
			 std::vector<int> dimensions, std::vector<bool> mask) {
	assert(dimensions.size() == mask.size() && !dimensions.empty());
	flattenTensor(dimensions, mask);
	if (mask.front()) {
		softmaxL(out, in, dimensions);
	}
	else {
		assert(!"Unsupported yet");
	}
}

//__global__ void softmaxKernel(float* out, const float* in)
//{
//	int id = blockIdx.x * blockDim.x + threadIdx.x;
//	out[id] = expf(in[id]) + expf(in[id] * 2) + expf(in[id] * 0.534);
//}
//
//void softmax(float* out, const float* in, const std::vector<int>& dimensions, bool opOnLeastSigDim) {
//	//temporary assumptions:
//	assert(opOnLeastSigDim == true);
//	assert(dimensions.size() == 2);
//
//	int height = dimensions[1];
//	int width = dimensions[0]; // least significant dimmension
//
//	dim3 blockSize(width);
//	dim3 gridSize(height);
//	softmaxKernel<<<gridSize, blockSize>>>(out, in);
//}

float softmax(CudaBuffer<float>& outHost, const CudaBuffer<float>& inHost,
	          const std::vector<int> dimensions, const std::vector<bool> mask) {
	CudaBuffer<float> outDev(outHost.size(), cudaMemoryTypeDevice);
	CudaBuffer<float> inDev(inHost.size(), cudaMemoryTypeDevice);
	inDev.copyFrom(inHost);

	CudaEvent start, stop;
	start.record();
	softmax(outDev(), inDev(), dimensions, mask);
	checkCudaError(cudaGetLastError());
	stop.record();
	stop.synchronize();
	float elapsedTime = start.elapsedTime(stop);

	checkCudaError(cudaDeviceSynchronize());
	outHost.copyFrom(outDev);

	return elapsedTime;
}

float softmaxMemCopy(CudaBuffer<float>& outHost, const CudaBuffer<float>& inHost,
	const std::vector<int> dimensions, const std::vector<bool> mask) {
	CudaBuffer<float> outDev(outHost.size(), cudaMemoryTypeDevice);
	CudaBuffer<float> inDev(inHost.size(), cudaMemoryTypeDevice);
	inDev.copyFrom(inHost);

	CudaEvent start, stop;
	start.record();
	outDev.copyFrom(inDev); // kernel replaced with memCopy
	checkCudaError(cudaGetLastError());
	stop.record();
	stop.synchronize();
	float elapsedTime = start.elapsedTime(stop);

	checkCudaError(cudaDeviceSynchronize());
	outHost.copyFrom(outDev);

	return elapsedTime;
}

size_t convIndex(size_t index, const std::vector<int>& dimensions, const std::vector<bool>& mask) {
	size_t newIndex = 0;
	size_t stride = 1;
	for (int i = 0; i < dimensions.size(); i++) {
		if (!mask[i]) {
			newIndex += (index % dimensions[i]) * stride;
			stride *= dimensions[i];
		}
		index /= dimensions[i];
	}
	return newIndex;
}

void softmaxRef(CudaBuffer<float>& outHost, const CudaBuffer<float>& inHost,
	            std::vector<int> dimensions, std::vector<bool> mask) {
	assert(dimensions.size() == mask.size());
	flattenTensor(dimensions, mask);

	size_t opSize = 1;
	size_t remSize = 1;
	for (int i = 0; i < dimensions.size(); i++) {
		assert(dimensions[i] > 1);
		if (mask[i])
			opSize *= dimensions[i];
		else
			remSize *= dimensions[i];
	}
	size_t size = opSize * remSize;
	assert(opSize > 1);

	std::vector<float> max(remSize);
	std::vector<double> acc(remSize);
	for (size_t i = 0; i < remSize; i++) {
		max[i] = std::numeric_limits<float>::min();
		acc[i] = 0;
	}

	for (size_t i = 0; i < size; i++) {
		size_t remIndex = convIndex(i, dimensions, mask);
		max[remIndex] = std::max(max[remIndex], inHost()[i]);
	}

	for (size_t i = 0; i < size; i++) {
		size_t remIndex = convIndex(i, dimensions, mask);
		float val = exp(inHost()[i] - max[remIndex]);
		acc[remIndex] += val;
		outHost()[i] = val;
	}

	for (size_t i = 0; i < size; i++) {
		size_t remIndex = convIndex(i, dimensions, mask);
		outHost()[i] /= acc[remIndex];
	}
}

double getTheoreticalTime(const std::vector<int>& dimensions) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	double clockRate = prop.memoryClockRate; // in kilohertz
	double busWidth = prop.memoryBusWidth; // in bits
	double bandwidth = (clockRate * 1000 * 2) * (busWidth / 8);

	size_t tensorSize = sizeof(float);
	for (auto d : dimensions)
		tensorSize *= d;

	double tensorAccessInSec = tensorSize / bandwidth;
	return tensorAccessInSec * 2 * 1000; // in milliseconds
}

void testSoftmax(const std::vector<int>& dimensions, const std::vector<bool>& mask) {
	if (dimensions.size() != mask.size() || mask.size() == 0)
		return;

	size_t size = 1;
	for (auto d : dimensions)
		size *= d;

	CudaBuffer<float> in(size, cudaMemoryTypeHost);
	in.fillWithRandom(-1000, 1000);

	CudaBuffer<float> out(size, cudaMemoryTypeHost);
	float timeMemCopy = softmaxMemCopy(out, in, dimensions, mask);
	float time = softmax(out, in, dimensions, mask);
	CudaBuffer<float> outRef(size, cudaMemoryTypeHost);
	softmaxRef(outRef, in, dimensions, mask);

	bool pass = out.approxEqual(outRef, 1e-5f);
	std::string name = "Softmax(";
	for (int i = 0; i < dimensions.size(); i++) {
		name += std::to_string(dimensions[i]);
		if (mask[i])
			name += "_s";
		if(i < dimensions.size() - 1)
			name += ", ";
	}
	name += ")";

	double theoreticalTime = getTheoreticalTime(dimensions);
	float memCopyEff = theoreticalTime / timeMemCopy;
	float kernelEff = theoreticalTime / time;
	std::ostringstream ss;
	ss << std::fixed << std::setprecision(2);
	ss << "  " << memCopyEff * 100 << "% " << kernelEff * 100 << "%";
	name += ss.str();

	printTestItem(name, time, pass);
}

void testSoftmax() {
	checkCudaError(cudaSetDevice(0));

	for (int i = 1; i <= 12; i++) {
		int size = 1 << i; // 2^i
		testSoftmax({ 1024 * 64, 8, 4, size, 2 }, { 1, 0, 0, 0, 0 });
	}

	//checkCudaError(cudaSetDevice(0));

	//for (int i = 1; i <= 12; i++) {
	//	int size = 1 << i; // 2^i
	//	testSoftmax({ 125, 64, 8, size, 2 }, { 1, 0, 1, 0, 1 });
	//}

	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceReset());
}
