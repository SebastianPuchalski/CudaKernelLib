#include "common.hpp"

#include <iostream>
#include <iomanip>

void printTestHeader() {
	std::cout << std::left << std::setw(55) << "Name:";
	std::cout << std::left << std::setw(15) << "Time(ms.):";
	std::cout << "Result:" << std::endl;
}

void printTestItem(const std::string& name, float time, bool pass) {
	assert(name.length() <= 50);
	std::cout << std::left << std::setw(55) << name;
	std::cout << std::left << std::setw(15) << time;
	std::cout << (pass ? "\033[32mPASS" : "\033[31mFAIL") << "\033[0m\n";
}

bool checkCudaError(cudaError_t cudaError, bool noThrow) {
	if (cudaError != cudaSuccess) {
		std::string errorString = "CUDA ERROR: ";
		errorString += std::string(cudaGetErrorString(cudaError));
		if (noThrow) {
			std::cerr << "\033[31m" << errorString << "\033[0m\n";
			return true;
		}
		throw std::runtime_error(errorString);
	}
	return false;
}

void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Name: " << prop.name << std::endl;
    std::cout << "UUID: ..." << std::endl;
    std::cout << "LUID: ..." << std::endl;
    std::cout << "LUID Device Node Mask: " << prop.luidDeviceNodeMask << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem << std::endl;
    std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << std::endl;
    std::cout << "Registers Per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Memory Pitch: " << prop.memPitch << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dimension: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max Grid Size: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
    std::cout << "Clock Rate: " << prop.clockRate << std::endl;
    std::cout << "Total Constant Memory: " << prop.totalConstMem << std::endl;
    std::cout << "Major Compute Capability: " << prop.major << std::endl;
    std::cout << "Minor Compute Capability: " << prop.minor << std::endl;
    std::cout << "Texture Alignment: " << prop.textureAlignment << std::endl;
    std::cout << "Texture Pitch Alignment: " << prop.texturePitchAlignment << std::endl;
    std::cout << "Device Overlap: " << prop.deviceOverlap << std::endl;
    std::cout << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Kernel Execution Timeout Enabled: " << prop.kernelExecTimeoutEnabled << std::endl;
    std::cout << "Integrated: " << prop.integrated << std::endl;
    std::cout << "Can Map Host Memory: " << prop.canMapHostMemory << std::endl;
    std::cout << "Compute Mode: " << prop.computeMode << std::endl;
    std::cout << "Max Texture 1D: " << prop.maxTexture1D << std::endl;
    std::cout << "Max Texture 1D Mipmap: " << prop.maxTexture1DMipmap << std::endl;
    std::cout << "Max Texture 1D Linear: " << prop.maxTexture1DLinear << std::endl;
    std::cout << "Max Texture 2D: " << prop.maxTexture2D[0] << " " << prop.maxTexture2D[1] << std::endl;
    std::cout << "Max Texture 2D Mipmap: " << prop.maxTexture2DMipmap[0] << " " << prop.maxTexture2DMipmap[1] << std::endl;
    std::cout << "Max Texture 2D Linear: " << prop.maxTexture2DLinear[0] << " " << prop.maxTexture2DLinear[1] << " " << prop.maxTexture2DLinear[2] << std::endl;
    std::cout << "Max Texture 2D Gather: " << prop.maxTexture2DGather[0] << " " << prop.maxTexture2DGather[1] << std::endl;
    std::cout << "Max Texture 3D: " << prop.maxTexture3D[0] << " " << prop.maxTexture3D[1] << " " << prop.maxTexture3D[2] << std::endl;
    std::cout << "Max Texture 3D Alt: " << prop.maxTexture3DAlt[0] << " " << prop.maxTexture3DAlt[1] << " " << prop.maxTexture3DAlt[2] << std::endl;
    std::cout << "Max Texture Cubemap: " << prop.maxTextureCubemap << std::endl;
    std::cout << "Max Texture 1D Layered: " << prop.maxTexture1DLayered[0] << " " << prop.maxTexture1DLayered[1] << std::endl;
    std::cout << "Max Texture 2D Layered: " << prop.maxTexture2DLayered[0] << " " << prop.maxTexture2DLayered[1] << " " << prop.maxTexture2DLayered[2] << std::endl;
    std::cout << "Max Texture Cubemap Layered: " << prop.maxTextureCubemapLayered[0] << " " << prop.maxTextureCubemapLayered[1] << std::endl;
    std::cout << "Max Surface 1D: " << prop.maxSurface1D << std::endl;
    std::cout << "Max Surface 2D: " << prop.maxSurface2D[0] << " " << prop.maxSurface2D[1] << std::endl;
    std::cout << "Max Surface 3D: " << prop.maxSurface3D[0] << " " << prop.maxSurface3D[1] << " " << prop.maxSurface3D[2] << std::endl;
    std::cout << "Max Surface 1D Layered: " << prop.maxSurface1DLayered[0] << " " << prop.maxSurface1DLayered[1] << std::endl;
    std::cout << "Max Surface 2D Layered: " << prop.maxSurface2DLayered[0] << " " << prop.maxSurface2DLayered[1] << " " << prop.maxSurface2DLayered[2] << std::endl;
    std::cout << "Max Surface Cubemap: " << prop.maxSurfaceCubemap << std::endl;
    std::cout << "Max Surface Cubemap Layered: " << prop.maxSurfaceCubemapLayered[0] << " " << prop.maxSurfaceCubemapLayered[1] << std::endl;
    std::cout << "Surface Alignment: " << prop.surfaceAlignment << std::endl;
    std::cout << "Concurrent Kernels: " << prop.concurrentKernels << std::endl;
    std::cout << "ECC Enabled: " << prop.ECCEnabled << std::endl;
    std::cout << "PCI Bus ID: " << prop.pciBusID << std::endl;
    std::cout << "PCI Device ID: " << prop.pciDeviceID << std::endl;
    std::cout << "PCI Domain ID: " << prop.pciDomainID << std::endl;
    std::cout << "TCC Driver: " << prop.tccDriver << std::endl;
    std::cout << "Async Engine Count: " << prop.asyncEngineCount << std::endl;
    std::cout << "Unified Addressing: " << prop.unifiedAddressing << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << std::endl;
    std::cout << "L2 Cache Size: " << prop.l2CacheSize << std::endl;
    std::cout << "Persisting L2 Cache Max Size: " << prop.persistingL2CacheMaxSize << std::endl;
    std::cout << "Max Threads Per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Stream Priorities Supported: " << prop.streamPrioritiesSupported << std::endl;
    std::cout << "Global L1 Cache Supported: " << prop.globalL1CacheSupported << std::endl;
    std::cout << "Local L1 Cache Supported: " << prop.localL1CacheSupported << std::endl;
    std::cout << "Shared Memory Per Multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Registers Per Multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
    std::cout << "Managed Memory: " << prop.managedMemory << std::endl;
    std::cout << "Is MultiGPU Board: " << prop.isMultiGpuBoard << std::endl;
    std::cout << "MultiGPU Board Group ID: " << prop.multiGpuBoardGroupID << std::endl;
    std::cout << "Host Native Atomic Supported: " << prop.hostNativeAtomicSupported << std::endl;
    std::cout << "Single To Double Precision Perf Ratio: " << prop.singleToDoublePrecisionPerfRatio << std::endl;
    std::cout << "Pageable Memory Access: " << prop.pageableMemoryAccess << std::endl;
    std::cout << "Concurrent Managed Access: " << prop.concurrentManagedAccess << std::endl;
    std::cout << "Compute Preemption Supported: " << prop.computePreemptionSupported << std::endl;
    std::cout << "Can Use Host Pointer For Registered Mem: " << prop.canUseHostPointerForRegisteredMem << std::endl;
    std::cout << "Cooperative Launch: " << prop.cooperativeLaunch << std::endl;
    std::cout << "Cooperative MultiDevice Launch: " << prop.cooperativeMultiDeviceLaunch << std::endl;
    std::cout << "Shared Memory Per Block Optin: " << prop.sharedMemPerBlockOptin << std::endl;
    std::cout << "Pageable Memory Access Uses Host Page Tables: " << prop.pageableMemoryAccessUsesHostPageTables << std::endl;
    std::cout << "Direct Managed Mem Access From Host: " << prop.directManagedMemAccessFromHost << std::endl;
    std::cout << "Max Blocks Per MultiProcessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Access Policy Max Window Size: " << prop.accessPolicyMaxWindowSize << std::endl;
    std::cout << "Reserved Shared Mem Per Block: " << prop.reservedSharedMemPerBlock << std::endl;
    std::cout << "Host Register Supported: " << prop.hostRegisterSupported << std::endl;
    std::cout << "Sparse CUDA Array Supported: " << prop.sparseCudaArraySupported << std::endl;
    std::cout << "Host Register Read Only Supported: " << prop.hostRegisterReadOnlySupported << std::endl;
    std::cout << "Timeline Semaphore Interop Supported: " << prop.timelineSemaphoreInteropSupported << std::endl;
    std::cout << "Memory Pools Supported: " << prop.memoryPoolsSupported << std::endl;
    std::cout << "GPUDirect RDMA Supported: " << prop.gpuDirectRDMASupported << std::endl;
    std::cout << "GPUDirect RDMA Flush Writes Options: " << prop.gpuDirectRDMAFlushWritesOptions << std::endl;
    std::cout << "GPUDirect RDMA Writes Ordering: " << prop.gpuDirectRDMAWritesOrdering << std::endl;
    std::cout << "Memory Pool Supported Handle Types: " << prop.memoryPoolSupportedHandleTypes << std::endl;
    std::cout << "Deferred Mapping CUDA Array Supported: " << prop.deferredMappingCudaArraySupported << std::endl;
    std::cout << "IPC Event Supported: " << prop.ipcEventSupported << std::endl;
    std::cout << "Cluster Launch: " << prop.clusterLaunch << std::endl;
    std::cout << "Unified Function Pointers: " << prop.unifiedFunctionPointers << std::endl;
}
