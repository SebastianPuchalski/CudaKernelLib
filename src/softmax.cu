#include "softmax.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <iomanip>

const int BASE_DIM_NUMBER = 2;
const int EXT_DIM_NUMBER = 4;
const int MAX_DIM_NUMBER = 64;
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
__global__ void softmaxKernelS(float* out, const float* in, int size0,
	                           int size2, int sizeEven, int addDimNumber, float alpha = 1) {
	const int WARP_SIZE = 32;

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

	float max = -FLT_MAX;
	for (int idxEven = threadIdx.z; idxEven < sizeEven; idxEven += blockDim.z) {
		float* lShared = shared;
		const float* lIn = in;
		if (addDimNumber) {
			lShared += idxEven * stride02;
			lIn += calcAddOffset(idxEven, idxOdd, addDimNumber) * stride0123;
		}
		for (int idx2 = threadIdx.y; idx2 < size2; idx2 += blockDim.y) {
			float* llShared = lShared + idx2 * stride0;
			const float* llIn = lIn + idx2 * stride01;
			for (int idx0 = threadIdx.x; idx0 < size0; idx0 += blockDim.x) {
				const float value = llIn[idx0];
				llShared[idx0] = value;
				max = fmaxf(value, max);
			}
		}
	}
	#pragma unroll
	for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
		max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, i));

	float sum = 0;
	for (int idxEven = threadIdx.z; idxEven < sizeEven; idxEven += blockDim.z) {
		float* lShared = shared + idxEven * stride02;
		for (int idx2 = threadIdx.y; idx2 < size2; idx2 += blockDim.y) {
			float* llShared = lShared + idx2 * stride0;
			for (int idx0 = threadIdx.x; idx0 < size0; idx0 += blockDim.x) {
				const float value = expf(llShared[idx0] - max);
				llShared[idx0] = value;
				sum += value;
			}
		}
	}
	#pragma unroll
	for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
		sum += __shfl_xor_sync(0xffffffff, sum, i);

	const float norm = alpha / sum;
	for (int idxEven = threadIdx.z; idxEven < sizeEven; idxEven += blockDim.z) {
		float* lShared = shared;
		float* lOut = out;
		if (addDimNumber) {
			lShared += idxEven * stride02;
			lOut += calcAddOffset(idxEven, idxOdd, addDimNumber) * stride0123;
		}
		for (int idx2 = threadIdx.y; idx2 < size2; idx2 += blockDim.y) {
			float* llShared = lShared + idx2 * stride0;
			float* llOut = lOut + idx2 * stride01;
			for (int idx0 = threadIdx.x; idx0 < size0; idx0 += blockDim.x)
				llOut[idx0] = llShared[idx0] * norm;
		}
	}
}

__global__ void softmaxKernelS(float* out, const float* in, int size0, float alpha = 1) {
	const int WARP_SIZE = 32;
	const int thread = threadIdx.x;

	extern __shared__ float shared[];

	const int64_t offset1 = blockIdx.x * size0;
	in += offset1;
	out += offset1;

	float max = -FLT_MAX;
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

	const float norm = alpha / sum;
	for (int i = thread; i < size0; i += WARP_SIZE)
		out[i] = shared[i] * norm;
}

int roundDownToPowerOfTwo(int number) {
	if (number <= 0) return 0;
	number |= number >> 1;
	number |= number >> 2;
	number |= number >> 4;
	number |= number >> 8;
	number |= number >> 16;
	return (number >> 1) + 1;
}

dim3 calcBlockDim(int blockSize, int size0, int size2, int sizeEven) {
	dim3 blockDim;
	blockDim.x = (size0 < 32) ? 8 : 32;
	if (size0 % 8 == 0) blockDim.x = 8;
	if (size0 % 16 == 0) blockDim.x = 16;
	if (size0 % 32 == 0) blockDim.x = 32;
	blockSize /= blockDim.x;
	blockDim.z = std::min(roundDownToPowerOfTwo(sizeEven), blockSize);
	blockSize /= blockDim.z;
	blockDim.y = std::min(roundDownToPowerOfTwo(size2), blockSize);
	blockSize /= blockDim.y;
	blockDim.x *= blockSize;
	return blockDim;
}

void softmaxS(float* out, const float* in, std::vector<int> dimensions) {
	const int WARP_SIZE = 32;

	assert(!dimensions.empty());
	int size0 = dimensions[0];
	assert(size0 > 1);
	int size1 = dimensions.size() > 1 ? dimensions[1] : 1;
	if (dimensions.size() <= BASE_DIM_NUMBER) {
		dim3 blockSize(WARP_SIZE);
		dim3 gridSize(size1);
		int sharedSize = size0 * sizeof(float);
		assert(sharedSize < 48 * 1024);
		softmaxKernelS<<<gridSize, blockSize, sharedSize>>>(out, in, size0);
		return;
	}
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
	assert(sizeOdd < 65536);
	dim3 gridDim(size1, size3, sizeOdd);
	int sharedSize = size0 * size2 * sizeEven * sizeof(float);
	dim3 blockDim = calcBlockDim(WARP_SIZE, size0, size2, sizeEven);

	if(addDimNum) {
		softmaxKernelS<<<gridDim, blockDim, sharedSize>>>
			(out, in, size0, size2, sizeEven, addDimNum);
	}
	else {
		if (size2 > 1 || size3 > 1) {
			softmaxKernelS<EXT_DIM_NUMBER><<<gridDim, blockDim, sharedSize>>>
				(out, in, size0, size2, sizeEven, addDimNum);
		}
		else {
			dim3 blockDim(WARP_SIZE);
			softmaxKernelS<BASE_DIM_NUMBER><<<gridDim, blockDim, sharedSize>>>
				(out, in, size0, size2, sizeEven, addDimNum);
		}
	}
}

template <int BLOCK_SIZE>
__inline__ __device__ void reduceMaxSumInBlock(float& max, float& sum, int thread) {
	assert(BLOCK_SIZE % 2 == 0);
	const int WARP_SIZE = 32;

	__shared__ float maxMem[BLOCK_SIZE];
	__shared__ float sumMem[BLOCK_SIZE];

	if (BLOCK_SIZE > WARP_SIZE) {
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
				sum1 *= expf(max1 - newMax);
				sum2 *= expf(max2 - newMax);
				maxMem[thread] = newMax;
				sumMem[thread] = sum1 + sum2;
			}
			__syncthreads();
		}

		if (thread < WARP_SIZE) {
			max = maxMem[thread];
			sum = sumMem[thread];
		}
	}

	if (thread < WARP_SIZE) {
		float newMax = max;
		#pragma unroll
		for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
			newMax = fmaxf(newMax, __shfl_xor_sync(0xffffffff, newMax, i));
		maxMem[thread] = newMax;

		sum *= expf(max - newMax);
		for (int i = WARP_SIZE / 2; i > 0; i >>= 1)
			sum += __shfl_xor_sync(0xffffffff, sum, i);
		sumMem[thread] = sum;
	}
	__syncthreads();
	max = *maxMem;
	sum = *sumMem;
}

template <int BLOCK_SIZE, int MAX_DIM_NUM = MAX_DIM_NUMBER>
__global__ void softmaxKernel(float* out, const float* in, int size0,
	                          int size2, int sizeEven, int addDimNumber, float alpha = 1) {
	const int WARP_SIZE = 32;

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

	const int64_t offset13 = idx1 * stride0 + idx3 * stride012;
	in += offset13;
	out += offset13;

	float max = -FLT_MAX;
	float sum = 0;

	for (int idxEven = threadIdx.z; idxEven < sizeEven; idxEven += blockDim.z) {
		const float* lIn = in;
		if (addDimNumber)
			lIn += calcAddOffset(idxEven, idxOdd, addDimNumber) * stride0123;
		for (int idx2 = threadIdx.y; idx2 < size2; idx2 += blockDim.y) {
			const float* llIn = lIn + idx2 * stride01;
			for (int idx0 = threadIdx.x; idx0 < size0; idx0 += blockDim.x) {
				const float value = llIn[idx0];
				const float newMax = fmaxf(max, value);
				const float fraction = expf(fminf(max, value) - newMax);
				sum = (value > max) ? sum * fraction + 1 : sum + fraction;
				max = newMax;
			}
		}
	}

	const int thread = threadIdx.x + (threadIdx.y + threadIdx.z * blockDim.y) * blockDim.x;
	reduceMaxSumInBlock<BLOCK_SIZE>(max, sum, thread);

	const float norm = alpha / sum;
	for (int idxEven = threadIdx.z; idxEven < sizeEven; idxEven += blockDim.z) {
		const float* lIn = in;
		float* lOut = out;
		if (addDimNumber) {
			const int64_t addOffset = calcAddOffset(idxEven, idxOdd, addDimNumber) * stride0123;
			lOut += addOffset;
			lIn += addOffset;
		}
		for (int idx2 = threadIdx.y; idx2 < size2; idx2 += blockDim.y) {
			const int offset2 = idx2 * stride01;
			const float* llIn = lIn + offset2;
			float* llOut = lOut + offset2;
			for (int idx0 = threadIdx.x; idx0 < size0; idx0 += blockDim.x)
				llOut[idx0] = expf(llIn[idx0] - max) * norm;
		}
	}
}

template <int BLOCK_SIZE>
__global__ void softmaxKernel(float* out, const float* in, int size0, float alpha = 1) {
	const int WARP_SIZE = 32;
	assert(WARP_SIZE == warpSize);
	const int thread = threadIdx.x;

	const int64_t offset1 = blockIdx.x * size0;
	in += offset1;
	out += offset1;

	float max = -FLT_MAX;
	float sum = 0;
	if (thread < size0) {
		max = in[thread];
		sum = 1;
	}
	for (int i = thread + BLOCK_SIZE; i < size0; i += BLOCK_SIZE) {
		const float value = in[i];
		const float newMax = fmaxf(max, value);
		const float fraction = expf(fminf(max, value) - newMax);
		sum = (value > max) ? sum * fraction + 1 : sum + fraction;
		max = newMax;
	}

	reduceMaxSumInBlock<BLOCK_SIZE>(max, sum, thread);

	const float norm = alpha / sum;
	for (int i = thread; i < size0; i += BLOCK_SIZE)
		out[i] = expf(in[i] - max) * norm;
}

void softmax(float* out, const float* in, std::vector<int> dimensions) {
	const int OP_SIZE_THRESHOLD = 4 * 1024;

	assert(!dimensions.empty());
	int size0 = dimensions[0];
	assert(size0 > 1);
	int size1 = dimensions.size() > 1 ? dimensions[1] : 1;
	if (dimensions.size() <= BASE_DIM_NUMBER) {
		dim3 gridDim(size1);
		if(size0 >= OP_SIZE_THRESHOLD) {
			const int BLOCK_SIZE = 1024;
			dim3 blockDim(BLOCK_SIZE);
			softmaxKernel<BLOCK_SIZE><<<gridDim, blockDim>>>(out, in, size0);
		}
		else {
			const int BLOCK_SIZE = 32;
			dim3 blockDim(BLOCK_SIZE);
			softmaxKernel<BLOCK_SIZE><<<gridDim, blockDim>>>(out, in, size0);
		}
		return;
	}
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
		checkCudaError(cudaGetLastError());
		for (int i = 0; i < dimensions.size(); i++) {
			assert(dimensions[i] > 1);
			if (i % 2 == 0)
				sizeEven *= dimensions[i];
			else
				sizeOdd *= dimensions[i];
		}
	}
	assert(sizeOdd < 65536);
	dim3 gridDim(size1, size3, sizeOdd);

	if (size0 * size2 * sizeEven >= OP_SIZE_THRESHOLD) {
		const int BLOCK_SIZE = 1024;
		dim3 blockDim = calcBlockDim(BLOCK_SIZE, size0, size2, sizeEven);
		if(addDimNum) {
			softmaxKernel<BLOCK_SIZE><<<gridDim, blockDim>>>
				(out, in, size0, size2, sizeEven, addDimNum);
		}
		else {
			if (size2 > 1 || size3 > 1) {
				softmaxKernel<BLOCK_SIZE, EXT_DIM_NUMBER><<<gridDim, blockDim>>>
					(out, in, size0, size2, sizeEven, addDimNum);
			}
			else {
				dim3 blockDim(BLOCK_SIZE);
				softmaxKernel<BLOCK_SIZE, BASE_DIM_NUMBER><<<gridDim, blockDim>>>
					(out, in, size0, size2, sizeEven, addDimNum);
			}
		}
	}
	else {
		const int BLOCK_SIZE = 32;
		dim3 blockDim = calcBlockDim(BLOCK_SIZE, size0, size2, sizeEven);
		if(addDimNum) {
			softmaxKernel<BLOCK_SIZE><<<gridDim, blockDim>>>
				(out, in, size0, size2, sizeEven, addDimNum);
		}
		else {
			if (size2 > 1 || size3 > 1) {
				softmaxKernel<BLOCK_SIZE, EXT_DIM_NUMBER><<<gridDim, blockDim>>>
					(out, in, size0, size2, sizeEven, addDimNum);
			}
			else {
				dim3 blockDim(BLOCK_SIZE);
				softmaxKernel<BLOCK_SIZE, BASE_DIM_NUMBER><<<gridDim, blockDim>>>
					(out, in, size0, size2, sizeEven, addDimNum);
			}
		}
	}
}

void flattenTensor(std::vector<int>& dimensions, std::vector<bool>& mask) {
	assert(dimensions.size() == mask.size() && !dimensions.empty());
	std::vector<int> newDimensions;
	std::vector<bool> newMask;
	bool previous = !mask.front();
	for (int i = 0; i < dimensions.size(); i++) {
		assert(dimensions[i] > 0);
		if(dimensions[i] == 1)
			continue;
		if (mask[i] == previous) {
			newDimensions.back() *= dimensions[i];
		}
		else {
			newDimensions.push_back(dimensions[i]);
			newMask.push_back(mask[i]);
		}
		previous = mask[i];
	}
	dimensions = newDimensions;
	mask = newMask;
}

void softmax(float* out, const float* in,
			 std::vector<int> dimensions, std::vector<bool> mask) {
	assert(dimensions.size() == mask.size() && !dimensions.empty());
	flattenTensor(dimensions, mask);
	if (mask.front()) {
		softmax(out, in, dimensions);
	}
	else {
		assert(!"Unsupported yet");
	}
}

float softmax(CudaBuffer<float>& outHost, const CudaBuffer<float>& inHost,
	          const std::vector<int> dimensions, const std::vector<bool> mask,
	          bool replaceWithMemcpy = false) {
	CudaBuffer<float> outDev(outHost.size(), cudaMemoryTypeDevice);
	CudaBuffer<float> inDev(inHost.size(), cudaMemoryTypeDevice);
	inDev.copyFrom(inHost);

	CudaEvent start, stop;
	start.record();
	if(replaceWithMemcpy)
		outDev.copyFrom(inDev);
	else
		softmax(outDev(), inDev(), dimensions, mask);
	checkCudaError(cudaGetLastError());
	stop.record();
	stop.synchronize();
	float elapsedTime = start.elapsedTime(stop);

	checkCudaError(cudaDeviceSynchronize());
	outHost.copyFrom(outDev);

	return elapsedTime;
}

size_t convIndex(size_t index, const std::vector<int>& dimensions,
	                           const std::vector<bool>& mask) {
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
		max[i] = std::numeric_limits<float>::lowest();
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

void testSoftmax(const std::vector<int>& dimensions,
	             const std::vector<bool>& mask) {
	if (dimensions.size() != mask.size() || mask.size() == 0)
		return;

	size_t size = 1;
	for (auto d : dimensions)
		size *= d;

	CudaBuffer<float> in(size, cudaMemoryTypeHost);
	in.fillWithRandom(-100, 100);

	CudaBuffer<float> out(size, cudaMemoryTypeHost);
	float memcpyTime = softmax(out, in, dimensions, mask, true);
	float kernelTime = softmax(out, in, dimensions, mask);
	CudaBuffer<float> outRef(size, cudaMemoryTypeHost);
	softmaxRef(outRef, in, dimensions, mask);

	bool pass = out.approxEqual(outRef);
	std::string name = "Softmax (";
	for (int i = 0; i < dimensions.size(); i++) {
		if (mask[i]) name += "[";
		name += std::to_string(dimensions[i]);
		if (mask[i]) name += "]";
		if(i < dimensions.size() - 1)
			name += ", ";
	}
	name += ")";

	float theoreticalTime = getBestMemoryAccessTime(size * sizeof(float));
	theoreticalTime *= 2 * 1000; // read and write in milliseconds
	float memcpyEff = theoreticalTime / memcpyTime;
	float kernelEff = theoreticalTime / kernelTime;
	std::ostringstream addInfo;
	addInfo << std::fixed << std::setprecision(0);
	addInfo << "memcpy: " << memcpyEff * 100 << "%  ";
	addInfo << "kernel: " << kernelEff * 100 << "%";

	printTestItem(name, pass, kernelTime, addInfo.str());
	checkCudaError(cudaDeviceReset());
}

void testSoftmax() {
	checkCudaError(cudaSetDevice(0));

	testSoftmax({ 32, 1024, 512 }, { 1, 0, 0 });
	testSoftmax({ 53, 1001, 1037 }, { 1, 0, 0 });
	testSoftmax({ 64, 1024, 1024 }, { 1, 0, 0 });
	testSoftmax({ 128, 1031, 250 }, { 1, 0, 0 });
	testSoftmax({ 256, 1024, 128 }, { 1, 0, 0 });
	testSoftmax({ 417, 1111, 55 }, { 1, 0, 0 });
	testSoftmax({ 1024, 1024, 32 }, { 1, 0, 0 });

	testSoftmax({ 1024, 2, 1024, 64 }, { 1, 1, 0, 0 });
	testSoftmax({ 1024, 4, 1024, 32 }, { 1, 1, 0, 0 });
	testSoftmax({ 1024, 8, 1024, 16 }, { 1, 1, 0, 0 });
	testSoftmax({ 999, 15, 1123, 8 }, { 1, 1, 0, 0 });
	testSoftmax({ 1024, 32, 1024, 4 }, { 1, 1, 0, 0 });
	testSoftmax({ 1024, 64, 1024, 2 }, { 1, 1, 0, 0 });
	testSoftmax({ 1024, 128, 256, 2 }, { 1, 1, 0, 0 });
	testSoftmax({ 1024, 512, 128, 2 }, { 1, 1, 0, 0 });

	testSoftmax({ 14, 256, 32, 512 }, { 1, 0, 1, 0 });
	testSoftmax({ 16, 256, 16, 512 }, { 1, 0, 1, 0 });
	testSoftmax({ 64, 256, 4, 512 }, { 1, 0, 1, 0 });
	testSoftmax({ 256, 256, 2, 256 }, { 1, 0, 1, 0 });
	testSoftmax({ 1024, 256, 2, 32 }, { 1, 0, 1, 0 });
	testSoftmax({ 4096, 64, 2, 32 }, { 1, 0, 1, 0 });
	testSoftmax({ 4096, 16, 16, 16 }, { 1, 0, 1, 0 });

	testSoftmax({ 244, 16, 4, 16, 4, 16 }, { 1, 0, 1, 0, 1, 0 });
	testSoftmax({ 256, 16, 16, 16, 4, 8 }, { 1, 0, 1, 0, 1, 0 });
	testSoftmax({ 256, 16, 64, 16, 4, 2 }, { 1, 0, 1, 0, 1, 0 });

	testSoftmax({ 4, 4, 16, 8, 2, 8, 2, 2, 2, 2, 2, 4, 2, 2, 2 }, { 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0 });
}
