#include "common.hpp"
#include "matrixMul.cuh"

#include <iostream>

void test() {
	testMatrixMul();
}

int main(int argc, char *argv[]) {
	printTestHeader();
	try {
		test();
	}
	catch (const std::exception& e) {
		std::cerr << "\033[31m" << e.what() << "\033[0m";
	}
	catch (...) {
		std::cerr << "\033[31m" << "Unidentified exception" << "\033[0m";
	}
	return 0;
}
