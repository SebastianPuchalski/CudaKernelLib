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
