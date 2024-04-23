#pragma once

#include <string>
#include <vector>

#include "cuda_runtime.h"

void printTestHeader();
void printTestItem(const std::string& name, float time, bool pass);

void fillVectorRandom(std::vector<float>& vec, float min = 0, float max = 1);
bool compareVectors(const std::vector<float>& vec1,
	                const std::vector<float>& vec2,
	                float maxDiff = 1e-6);

bool checkCudaError(cudaError_t cudaError, bool noThrow = false);

template <typename T>
class CudaBuffer {
	int elNumber;
	T* data;
public:
	CudaBuffer(int size): elNumber(size), data(nullptr) {
		checkCudaError(cudaMalloc((void**)&data, size * sizeof(T)));
	}
	virtual ~CudaBuffer() {
		if(data)
			checkCudaError(cudaFree(data), true);
	}
	CudaBuffer(const CudaBuffer&) = delete;
	CudaBuffer& operator=(const CudaBuffer&) = delete;
	CudaBuffer(CudaBuffer&&) = delete;
	CudaBuffer& operator=(CudaBuffer&&) = delete;
	int size() {
		return elNumber;
	}
	int dataSize() {
		return elNumber * sizeof(T);
	}
	T* operator()() {
		return data;
	}
	// TODO:
	// - not only device but also host and unified memory
	// - operator= for memcpy
	// - shared_ptr typedef
	// - fill with zeros
	// - fill with random values
	// - operator== (for host only?)
};

class CudaEvent {
	cudaEvent_t event;
public:
	CudaEvent(): event(0) {
		checkCudaError(cudaEventCreate(&event));
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
	// TODO: shared_ptr typedef
};
