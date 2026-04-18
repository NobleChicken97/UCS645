#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

__global__ void array_sum_kernel(float *input, float *block_sums, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        sdata[tid] = input[i];
    else
        sdata[tid] = 0.0f;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    printf("Array Sum using CUDA\n");
    printf("Array size: %d elements\n\n", N);

    float *h_input = (float *)malloc(size);
    if (!h_input) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
        cpu_sum += h_input[i];
    }
    printf("CPU sum (for verification): %.2f\n", cpu_sum);

    float *d_input;
    CUDA_CHECK(cudaMalloc((void **)&d_input, size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    printf("Block size: %d\n", blockSize);
    printf("Grid size : %d\n\n", gridSize);

    float *d_block_sums;
    CUDA_CHECK(cudaMalloc((void **)&d_block_sums, gridSize * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    array_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_block_sums, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    float *h_block_sums = (float *)malloc(gridSize * sizeof(float));
    if (!h_block_sums) {
        fprintf(stderr, "Host malloc failed for block sums\n");
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    float gpu_sum = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        gpu_sum += h_block_sums[i];
    }

    printf("GPU sum: %.2f\n", gpu_sum);
    printf("Kernel time: %.4f ms\n", kernel_ms);
    printf("Difference: %.6e\n", fabs(gpu_sum - cpu_sum));

    if (fabs(gpu_sum - cpu_sum) < 1.0f) {
        printf("PASSED - results match\n");
    } else {
        printf("FAILED - results dont match\n");
    }

    cudaFree(d_input);
    cudaFree(d_block_sums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_input);
    free(h_block_sums);

    return 0;
}
