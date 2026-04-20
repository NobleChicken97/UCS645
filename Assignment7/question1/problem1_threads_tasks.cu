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

__global__ void iterative_sum_kernel(const int *input, unsigned long long *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int limit = input[idx];
        unsigned long long sum = 0;
        for (int i = 1; i <= limit; i++) {
            sum += (unsigned long long)i;
        }
        output[idx] = sum;
    }
}

__global__ void formula_sum_kernel(const int *input, unsigned long long *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long value = (unsigned long long)input[idx];
        output[idx] = (value * (value + 1ULL)) / 2ULL;
    }
}

int main() {
    const int N = 1024;
    const size_t intBytes = N * sizeof(int);
    const size_t outBytes = N * sizeof(unsigned long long);

    int *h_input = (int *)malloc(intBytes);
    unsigned long long *h_iter = (unsigned long long *)malloc(outBytes);
    unsigned long long *h_formula = (unsigned long long *)malloc(outBytes);

    if (!h_input || !h_iter || !h_formula) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1;
    }

    int *d_input = NULL;
    unsigned long long *d_iter = NULL;
    unsigned long long *d_formula = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_input, intBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_iter, outBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_formula, outBytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, intBytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t startIter, stopIter, startFormula, stopFormula;
    CUDA_CHECK(cudaEventCreate(&startIter));
    CUDA_CHECK(cudaEventCreate(&stopIter));
    CUDA_CHECK(cudaEventCreate(&startFormula));
    CUDA_CHECK(cudaEventCreate(&stopFormula));

    CUDA_CHECK(cudaEventRecord(startIter));
    iterative_sum_kernel<<<gridSize, blockSize>>>(d_input, d_iter, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stopIter));
    CUDA_CHECK(cudaEventSynchronize(stopIter));

    CUDA_CHECK(cudaEventRecord(startFormula));
    formula_sum_kernel<<<gridSize, blockSize>>>(d_input, d_formula, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stopFormula));
    CUDA_CHECK(cudaEventSynchronize(stopFormula));

    float iterMs = 0.0f;
    float formulaMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&iterMs, startIter, stopIter));
    CUDA_CHECK(cudaEventElapsedTime(&formulaMs, startFormula, stopFormula));

    CUDA_CHECK(cudaMemcpy(h_iter, d_iter, outBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_formula, d_formula, outBytes, cudaMemcpyDeviceToHost));

    int mismatchCount = 0;
    for (int i = 0; i < N; i++) {
        if (h_iter[i] != h_formula[i]) {
            mismatchCount++;
        }
    }

    printf("Problem 1: all threads doing different N based sum tasks\n");
    printf("N value: %d\n", N);
    printf("Block size: %d\n", blockSize);
    printf("Grid size: %d\n\n", gridSize);

    printf("Final sum for first %d integers iterative : %llu\n", N, h_iter[N - 1]);
    printf("Final sum for first %d integers formula   : %llu\n", N, h_formula[N - 1]);
    printf("Iterative kernel time (ms): %.6f\n", iterMs);
    printf("Formula kernel time (ms)  : %.6f\n", formulaMs);

    if (mismatchCount == 0) {
        printf("Check status: PASS both methods giving same output for all threads\n");
    } else {
        printf("Check status: FAIL mismatches found = %d\n", mismatchCount);
    }

    printf("\nSample outputs from few threads\n");
    for (int i = 0; i < 10; i++) {
        printf("thread %4d n=%4d iterative=%10llu formula=%10llu\n",
               i, h_input[i], h_iter[i], h_formula[i]);
    }

    CUDA_CHECK(cudaEventDestroy(startIter));
    CUDA_CHECK(cudaEventDestroy(stopIter));
    CUDA_CHECK(cudaEventDestroy(startFormula));
    CUDA_CHECK(cudaEventDestroy(stopFormula));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_iter));
    CUDA_CHECK(cudaFree(d_formula));

    free(h_input);
    free(h_iter);
    free(h_formula);

    return 0;
}
