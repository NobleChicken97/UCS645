#include <math.h>
#include <stdio.h>
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

#define N (1 << 20)

__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

__global__ void vector_add_symbol_kernel(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

int main() {
    static float h_A[N];
    static float h_B[N];
    static float h_C[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(i % 100) * 0.5f;
        h_B[i] = (float)(i % 50) * 1.5f;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float)));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vector_add_symbol_kernel<<<gridSize, blockSize>>>(N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));

    CUDA_CHECK(cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float)));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-5f) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at %d got %.6f expected %.6f\n", i, h_C[i], expected);
            }
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    double theoreticalGbps = ((double)prop.memoryClockRate * (double)prop.memoryBusWidth * 2.0) / 1e6;
    double theoreticalGBs = theoreticalGbps / 8.0;

    double readBytes = (double)N * 2.0 * sizeof(float);
    double writeBytes = (double)N * 1.0 * sizeof(float);
    double totalBytes = readBytes + writeBytes;
    double kernelSec = (double)kernelMs / 1000.0;
    double measuredGBs = totalBytes / kernelSec / 1e9;

    printf("Problem 3: vector add with static global device symbols\n");
    printf("N value                             : %d\n", N);
    printf("Kernel time (ms)                    : %.6f\n", kernelMs);
    printf("Check status                        : %s\n", (errors == 0) ? "PASS" : "FAIL");
    printf("GPU name                            : %s\n", prop.name);
    printf("Memory clock rate (kHz)             : %d\n", prop.memoryClockRate);
    printf("Memory bus width (bits)             : %d\n", prop.memoryBusWidth);
    printf("Theoretical bandwidth (GB/s)        : %.3f\n", theoreticalGBs);
    printf("Measured bandwidth from kernel (GB/s): %.3f\n", measuredGBs);

    if (theoreticalGBs > 0.0) {
        printf("Measured/theoretical ratio          : %.4f\n", measuredGBs / theoreticalGBs);
    }

    printf("\nProfiling command\n");
    printf("Try: nvprof ./question3/problem3_vector_add_profile\n");
    printf("If nvprof is missing in your CUDA version then use: nsys profile ./question3/problem3_vector_add_profile\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
