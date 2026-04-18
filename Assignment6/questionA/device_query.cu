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

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found on this system\n");
        return 1;
    }

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        int coreClockKHz = 0;
        int memClockKHz = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&coreClockKHz, cudaDevAttrClockRate, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev));

        printf("========== Device %d: %s ==========\n\n", dev, prop.name);

        printf("--- Compute Capability ---\n");
        printf("Compute Capability       : %d.%d\n", prop.major, prop.minor);
        printf("\n");

        printf("--- Processor Info ---\n");
        printf("Number of SMs            : %d\n", prop.multiProcessorCount);
        printf("Max threads per SM       : %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max threads per block    : %d\n", prop.maxThreadsPerBlock);
        printf("Warp size                : %d\n", prop.warpSize);
        printf("\n");

        printf("--- Max Block Dimensions ---\n");
        printf("Max block dim x          : %d\n", prop.maxThreadsDim[0]);
        printf("Max block dim y          : %d\n", prop.maxThreadsDim[1]);
        printf("Max block dim z          : %d\n", prop.maxThreadsDim[2]);
        printf("\n");

        printf("--- Max Grid Dimensions ---\n");
        printf("Max grid dim x           : %d\n", prop.maxGridSize[0]);
        printf("Max grid dim y           : %d\n", prop.maxGridSize[1]);
        printf("Max grid dim z           : %d\n", prop.maxGridSize[2]);
        printf("\n");

        printf("--- Memory Info ---\n");
        printf("Global memory            : %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("Shared memory per block  : %lu KB\n", prop.sharedMemPerBlock / 1024);
        printf("Constant memory          : %lu KB\n", prop.totalConstMem / 1024);
        printf("Registers per block      : %d\n", prop.regsPerBlock);
        printf("\n");

        printf("--- Other Details ---\n");
        printf("Clock rate               : %.2f GHz\n", coreClockKHz / 1e6);
        printf("Memory clock rate        : %.2f GHz\n", memClockKHz / 1e6);
        printf("Memory bus width         : %d bits\n", prop.memoryBusWidth);
        printf("L2 cache size            : %d KB\n", prop.l2CacheSize / 1024);
        printf("Max threads per SM       : %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Concurrent kernels       : %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("ECC enabled              : %s\n", prop.ECCEnabled ? "Yes" : "No");

        if (prop.major > 1 || (prop.major == 1 && prop.minor >= 3)) {
            printf("Double precision         : Supported\n");
        } else {
            printf("Double precision         : Not supported\n");
        }

        printf("\n");
    }

    return 0;
}
