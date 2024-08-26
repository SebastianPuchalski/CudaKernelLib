# CudaKernelLib

Educational project that will contain a collection of CUDA kernels. I am focusing primarily on kernels for deep learning purposes.

### Softmax
General and fast implementation of softmax for tensors of arbitrary dimensions (N-D) and sizes. It performs faster than cudaMemcpy for k in the range from 64 to 64000, where k is the product of dimensions for which softmax is applied.

| Size | Kernel time | Kernel efficiency | cudaMemcpy efficiency |
|---|---|---|---|
| ([53], 1001, 1037) | 1.664ms | 92% | 87% |
| ([256], 1024, 128) | 1.005ms | 93% | 90% |
| ([1024], [4], 1024, 32) | 4.166ms | 89% | 86% |
| ([1024], [64], 1024, 2) | 4.330ms | 86% | 85% |
| ([1024], [512], 128, 2) | 6.216ms | 60% | 86% |
| ([14], 256, [32], 512) | 1.813ms | 90% | 87% |
| ([256], 16, [16], 16, [4], 8) | 1.023ms | 91% | 90% |
| ([4], [4], [16], 8, 2, [8], [2], 2, 2, 2, 2, [4], 2, [2], 2) | 1.031ms | 90% | 90% |

Efficiency = theoretical data transmission time / measured time

**Constraints:** Softmax must be applied at least along the least significant dimension.

### Matrix multiplication

The implementation of matrix multiplication outperforms (3-10%) cuBLAS on the RTX 4060 Ti 16GB for matrices of common sizes:

| Size | matrixMulFast | cublasSgemm |
|---|---|---|
| 1024x1024x1024 | 0.185ms | 0.203ms |
| 2048x2048x2048 | 1.230ms | 1.289ms |
| 4096x4096x4096 | 9.642ms | 9.932ms |

**Constraints:** Each matrix column and row size must be a multiple of 32 or 64.

### GeGLU

Simple implementation of GeGLU.

| Size (batch, dim) | Time | Efficiency |
|---|---|---|
| (256, 1024*128) | 0.748ms | 93% |
| (1024, 1024*128) | 3.146ms | 89% |

**Constraints:** None.
