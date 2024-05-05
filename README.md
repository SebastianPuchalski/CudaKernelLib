# CudaKernelLib

Educational project that will contain a collection of CUDA kernels. I am focusing primarily on kernels for deep learning purposes.

### Matrix multiplication

The implementation of matrix multiplication outperforms cuBLAS on the RTX 4060 Ti 16GB for (squared) matrices of common sizes:

|   | 1024 | 2048 | 4096 |
|---|---|---|
| cublasSgemm   | 0.203ms | 1.289ms | 9.932ms |
| matrixMulFast | 0.185ms | 1.230ms | 9.642ms |
