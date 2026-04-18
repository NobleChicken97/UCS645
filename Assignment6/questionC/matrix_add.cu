#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err__));                     \
            return 1;                                                         \
        }                                                                     \
    } while (0)

__global__ void matrix_add_kernel(int *A, int *B, int *C, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int rows = 4096;
    int cols = 4096;
    int N = rows * cols;
    size_t size = N * sizeof(int);

    printf("Matrix Addition using CUDA\n");
    printf("Matrix size: %d x %d\n\n", rows, cols);

    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        h_A[i] = i % 100;
        h_B[i] = (i * 2) % 100;
    }

    int *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x,
                 (rows + blockDim.y - 1) / blockDim.y);

    printf("Block dimensions: %d x %d\n", blockDim.x, blockDim.y);
    printf("Grid dimensions : %d x %d\n\n", gridDim.x, gridDim.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    matrix_add_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        int expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at index %d: got %d, expected %d\n",
                       i, h_C[i], expected);
            }
        }
    }

    if (errors == 0) {
        printf("PASSED - all elements correct\n");
    } else {
        printf("FAILED - %d errors found\n", errors);
    }

    printf("Kernel time: %.4f ms\n\n", milliseconds);

    printf("--- Analysis ---\n");
    printf("Total elements (N x N)          : %d\n", N);
    printf("Floating point ops (additions)  : %d\n", N);
    printf("Global memory reads             : %d (N from A + N from B = 2N)\n", 2 * N);
    printf("Global memory writes            : %d (N to C)\n", N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
