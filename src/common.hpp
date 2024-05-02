#pragma once

#include <string>
#include <vector>
#include <cassert>
#include <random>

#include "cuda_runtime.h"

void printTestHeader();
void printTestItem(const std::string& name, float time, bool pass);

bool checkCudaError(cudaError_t cudaError, bool noThrow = false);

void printDevicesProperties();

template <typename T>
class CudaBuffer {
	int elNumber;
	cudaMemoryType memType;
	T* data;

public:
	CudaBuffer(int size, cudaMemoryType type): elNumber(size), memType(type), data(nullptr) {
		switch (type) {
		case cudaMemoryTypeHost:
			data = new T[size];
			break;
		case cudaMemoryTypeDevice:
			checkCudaError(cudaMalloc(&data, size * sizeof(T)));
			break;
		case cudaMemoryTypeManaged:
			checkCudaError(cudaMallocManaged(&data, size * sizeof(T)));
			break;
		default:
			assert(!"Wrong buffer type!");
		};
	}

	virtual ~CudaBuffer() {
		if (data) {
			switch (memType) {
			case cudaMemoryTypeHost:
				delete[] data;
				break;
			case cudaMemoryTypeDevice:
			case cudaMemoryTypeManaged:
				checkCudaError(cudaFree(data), true);
				break;
			};
		}
	}

	CudaBuffer(const CudaBuffer&) = delete;
	CudaBuffer& operator=(const CudaBuffer&) = delete;
	CudaBuffer(CudaBuffer&&) = delete;
	CudaBuffer& operator=(CudaBuffer&&) = delete;

	bool operator==(const CudaBuffer& rhs) const {
		assert(memType != cudaMemoryTypeDevice);
		if (elNumber != rhs.elNumber)
			return false;
		for (int i = 0; i < elNumber; i++) {
			if (data[i] != rhs.data[i])
				return false;
		}
		return true;
	}

	bool approxEqual(const CudaBuffer<float>& rhs, float maxRelDiff = 1e-6) const {
		assert(memType != cudaMemoryTypeDevice);
		if (elNumber != rhs.elNumber)
			return false;
		for (int i = 0; i < elNumber; i++) {
			float a = data[i];
			float b = rhs.data[i];
			float diff = abs(a - b);
			float mean = (abs(a) + abs(b)) / 2.0f;
			if (diff / (mean + FLT_MIN) > maxRelDiff)
				return false;
		}
		return true;
	}

	T* operator()() const {
		return data;
	}

	int size() const {
		return elNumber;
	}

	int dataSize() const {
		return elNumber * sizeof(T);
	}

	cudaMemoryType type() const {
		return memType;
	}

	void fillWithZeros() {
		assert(memType != cudaMemoryTypeDevice);
		for (int i = 0; i < elNumber; i++) {
			data[i] = 0;
		}
	}

	CudaBuffer& copyFrom(const CudaBuffer& rhs) {
		assert(dataSize() == rhs.dataSize());
		cudaMemcpyKind kind;
		if (rhs.memType == cudaMemoryTypeManaged || memType == cudaMemoryTypeManaged) {
			assert(memType == cudaMemoryTypeManaged);
			assert(rhs.memType == cudaMemoryTypeManaged);
			kind = cudaMemcpyDefault;
		}
		else {
			if (rhs.memType == cudaMemoryTypeHost && memType == cudaMemoryTypeHost)
				kind = cudaMemcpyHostToHost;
			else if (rhs.memType == cudaMemoryTypeHost && memType == cudaMemoryTypeDevice)
				kind = cudaMemcpyHostToDevice;
			else if (rhs.memType == cudaMemoryTypeDevice && memType == cudaMemoryTypeHost)
				kind = cudaMemcpyDeviceToHost;
			else if (rhs.memType == cudaMemoryTypeDevice && memType == cudaMemoryTypeDevice)
				kind = cudaMemcpyDeviceToDevice;
			else
				assert(!"Incompatible buffer types!");
		}
		checkCudaError(cudaMemcpy(data, rhs.data, dataSize(), kind));
		return *this;
	}

	void fillWithRandom(float min = 0, float max = 1) {
		assert(memType != cudaMemoryTypeDevice);
		static const unsigned seed = 3870562;
		static std::mt19937 gen(seed);
		std::uniform_real_distribution<float> dist(min, max);
		for (int i = 0; i < elNumber; i++) {
			data[i] = dist(gen);
		}
	}
};

class CudaEvent {
	cudaEvent_t event;

public:
	CudaEvent(): event(0) {
		checkCudaError(cudaEventCreate(&event));
	}

	CudaEvent(unsigned int flags) : event(0) {
		checkCudaError(cudaEventCreateWithFlags(&event, flags));
	}

	virtual ~CudaEvent() {
		if(event)
			checkCudaError(cudaEventDestroy(event), true);
	}

	CudaEvent(const CudaEvent&) = delete;
	CudaEvent& operator=(const CudaEvent&) = delete;
	CudaEvent(CudaEvent&&) = delete;
	CudaEvent& operator=(CudaEvent&&) = delete;

	cudaEvent_t operator()() {
		return event;
	}

	void record(cudaStream_t stream = (cudaStream_t)0) {
		checkCudaError(cudaEventRecord(event, stream));
	}

	void recordWithFlags(cudaStream_t stream = (cudaStream_t)0, unsigned int flags = 0) {
		checkCudaError(cudaEventRecordWithFlags(event, stream, flags));
	}

	void synchronize() {
		checkCudaError(cudaEventSynchronize(event));
	}

	void query() {
		checkCudaError(cudaEventQuery(event));
	}

	float elapsedTime(CudaEvent& stop) {
		float elapsedTime;
		checkCudaError(cudaEventElapsedTime(&elapsedTime, event, stop()));
		return elapsedTime;
	}
};
