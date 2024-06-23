#include "common.hpp"
#include "softmax.cuh"
#include "matrixMul.cuh"
#include "geglu.cuh"

#include <iostream>

void test() {
	/*testSoftmax();
	testMatrixMul();*/
	testGeGLU();
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
