#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <omp.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err__));                     \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static inline int min_int(int a, int b) {
    return (a < b) ? a : b;
}

static void merge_segments(const int *src, int *dst, int left, int mid, int right) {
    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right) {
        if (src[i] <= src[j]) {
            dst[k++] = src[i++];
        } else {
            dst[k++] = src[j++];
        }
    }
    while (i < mid) {
        dst[k++] = src[i++];
    }
    while (j < right) {
        dst[k++] = src[j++];
    }
}

static void pipelined_merge_sort_cpu(int *data, int n) {
    int *temp = (int *)malloc(n * sizeof(int));
    int *src = data;
    int *dst = temp;

    for (int width = 1; width < n; width <<= 1) {
        int totalMerges = (n + (2 * width) - 1) / (2 * width);

#pragma omp parallel for schedule(static)
        for (int m = 0; m < totalMerges; m++) {
            int left = m * 2 * width;
            int mid = min_int(left + width, n);
            int right = min_int(left + 2 * width, n);
            merge_segments(src, dst, left, mid, right);
        }

        int *swapTmp = src;
        src = dst;
        dst = swapTmp;
    }

    if (src != data) {
        memcpy(data, src, n * sizeof(int));
    }

    free(temp);
}

__global__ void merge_pass_kernel(const int *src, int *dst, int width, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left = tid * (2 * width);

    if (left >= n) {
        return;
    }

    int mid = left + width;
    int right = left + (2 * width);

    if (mid > n) {
        mid = n;
    }
    if (right > n) {
        right = n;
    }

    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right) {
        if (src[i] <= src[j]) {
            dst[k++] = src[i++];
        } else {
            dst[k++] = src[j++];
        }
    }
    while (i < mid) {
        dst[k++] = src[i++];
    }
    while (j < right) {
        dst[k++] = src[j++];
    }
}

int main() {
    const int N = 1000;
    const size_t bytes = N * sizeof(int);

    int *h_input = (int *)malloc(bytes);
    int *h_cpu_sorted = (int *)malloc(bytes);
    int *h_gpu_sorted = (int *)malloc(bytes);

    if (!h_input || !h_cpu_sorted || !h_gpu_sorted) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % 10000;
    }

    memcpy(h_cpu_sorted, h_input, bytes);
    memcpy(h_gpu_sorted, h_input, bytes);

    auto cpuStart = std::chrono::high_resolution_clock::now();
    pipelined_merge_sort_cpu(h_cpu_sorted, N);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

    int *d_src = NULL;
    int *d_dst = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_src, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, bytes));
    CUDA_CHECK(cudaMemcpy(d_src, h_gpu_sorted, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t totalStart, totalStop, passStart, passStop;
    CUDA_CHECK(cudaEventCreate(&totalStart));
    CUDA_CHECK(cudaEventCreate(&totalStop));
    CUDA_CHECK(cudaEventCreate(&passStart));
    CUDA_CHECK(cudaEventCreate(&passStop));

    float kernelOnlyMs = 0.0f;
    CUDA_CHECK(cudaEventRecord(totalStart));

    const int threads = 256;
    for (int width = 1; width < N; width <<= 1) {
        int totalMerges = (N + (2 * width) - 1) / (2 * width);
        int blocks = (totalMerges + threads - 1) / threads;

        CUDA_CHECK(cudaEventRecord(passStart));
        merge_pass_kernel<<<blocks, threads>>>(d_src, d_dst, width, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(passStop));
        CUDA_CHECK(cudaEventSynchronize(passStop));

        float passMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&passMs, passStart, passStop));
        kernelOnlyMs += passMs;

        int *swapPtr = d_src;
        d_src = d_dst;
        d_dst = swapPtr;
    }

    CUDA_CHECK(cudaMemcpy(h_gpu_sorted, d_src, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(totalStop));
    CUDA_CHECK(cudaEventSynchronize(totalStop));

    float gpuTotalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTotalMs, totalStart, totalStop));

    int mismatch = 0;
    for (int i = 0; i < N; i++) {
        if (h_cpu_sorted[i] != h_gpu_sorted[i]) {
            mismatch++;
        }
    }

    bool sortedOK = true;
    for (int i = 1; i < N; i++) {
        if (h_gpu_sorted[i] < h_gpu_sorted[i - 1]) {
            sortedOK = false;
            break;
        }
    }

    printf("Problem 2: merge sort size n = %d\n", N);
    printf("CPU pipelined merge sort time (ms) : %.6f\n", cpuMs);
    printf("CUDA merge sort kernel only (ms)   : %.6f\n", kernelOnlyMs);
    printf("CUDA merge sort total time (ms)    : %.6f\n", gpuTotalMs);

    if (gpuTotalMs > 0.0f) {
        printf("Speedup cpu/gpu_total              : %.6f x\n", cpuMs / gpuTotalMs);
    }

    if (mismatch == 0 && sortedOK) {
        printf("Check status: PASS sorted output of both methods is same\n");
    } else {
        printf("Check status: FAIL mismatch count = %d sortedOK = %d\n", mismatch, sortedOK ? 1 : 0);
    }

    printf("\nFirst 20 sorted values\n");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_gpu_sorted[i]);
    }
    printf("\n");

    CUDA_CHECK(cudaEventDestroy(totalStart));
    CUDA_CHECK(cudaEventDestroy(totalStop));
    CUDA_CHECK(cudaEventDestroy(passStart));
    CUDA_CHECK(cudaEventDestroy(passStop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    free(h_input);
    free(h_cpu_sorted);
    free(h_gpu_sorted);

    return 0;
}
