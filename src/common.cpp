#include "common.hpp"

#include <cassert>
#include <iostream>
#include <iomanip>
#include <random>

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

void fillVectorRandom(std::vector<float>& vec, float min, float max) {
	static const unsigned seed = 3870562;
	static std::mt19937 gen(seed);
	std::uniform_real_distribution<float> dist(min, max);
	for (auto& number : vec) {
		number = dist(gen);
	}
}

bool compareVectors(const std::vector<float>& vec1,
	                const std::vector<float>& vec2,
	                float maxDiff) {
	if (vec1.size() != vec2.size())
		return false;
	for (int i = 0; i < vec1.size(); i++) {
		float a = vec1[i];
		float b = vec2[i];
		float diff = abs(a - b);
		float mean = (abs(a) + abs(b)) / 2.0f;
		if (diff / (mean + FLT_MIN) > maxDiff)
			return false;
	}
	return true;
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
