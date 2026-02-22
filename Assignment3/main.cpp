#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <sys/resource.h>
#include <omp.h>

extern void correlate(int ny, int nx, const float *data, float *result);

void generate_data(float *data, int ny, int nx) {
    unsigned int seed = 42;
    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
            data[y * nx + x] = (float)rand_r(&seed) / RAND_MAX * 2.0f - 1.0f;
}

int main() {
    int test_ny[] = {300, 600, 1000};
    int test_nx[] = {3000, 5000, 8000};
    int num_tests = 3;

    printf("Physical Cores: 4\n");
    printf("Logical Cores : 8 (Hyper-Threading)\n");
    printf("Max OpenMP threads: %d\n\n", omp_get_max_threads());

    printf("=== Thread Scaling ===\n\n");
    printf("%-8s | %-8s | %-8s | %-14s | %-14s | %-10s\n",
           "ny", "nx", "Threads", "Seq Time (s)", "Par Time (s)", "Speedup");
    printf("---------|----------|----------|----------------|----------------|----------\n");

    for (int s = 0; s < num_tests; s++) {
        int ny = test_ny[s];
        int nx = test_nx[s];

        float *data = (float *)malloc(ny * nx * sizeof(float));
        float *result = (float *)malloc(ny * ny * sizeof(float));

        generate_data(data, ny, nx);

        omp_set_num_threads(1);
        memset(result, 0, ny * ny * sizeof(float));
        double t1 = omp_get_wtime();
        correlate(ny, nx, data, result);
        double t2 = omp_get_wtime();
        double seq_time = t2 - t1;

        for (int threads = 1; threads <= 10; threads++) {
            omp_set_num_threads(threads);
            memset(result, 0, ny * ny * sizeof(float));

            double t3 = omp_get_wtime();
            correlate(ny, nx, data, result);
            double t4 = omp_get_wtime();
            double par_time = t4 - t3;
            double speedup = seq_time / par_time;

            printf("%-8d | %-8d | %-8d | %-14.6f | %-14.6f | %-10.2f\n",
                   ny, nx, threads, seq_time, par_time, speedup);
        }
        printf("---------|----------|----------|----------------|----------------|----------\n");

        free(data);
        free(result);
    }

    printf("\n=== Performance Stats (ny=800, nx=5000, 4 threads) ===\n\n");

    {
        int ny = 800, nx = 5000;
        float *data = (float *)malloc(ny * nx * sizeof(float));
        float *result = (float *)malloc(ny * ny * sizeof(float));

        generate_data(data, ny, nx);

        omp_set_num_threads(4);

        struct rusage usage_before, usage_after;
        getrusage(RUSAGE_SELF, &usage_before);

        clock_t cpu_start = clock();
        double wall_start = omp_get_wtime();

        correlate(ny, nx, data, result);

        double wall_end = omp_get_wtime();
        clock_t cpu_end = clock();

        getrusage(RUSAGE_SELF, &usage_after);

        double wall_time = wall_end - wall_start;
        double task_clock_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0;
        double cpus_used = (task_clock_ms / 1000.0) / wall_time;
        long vol_cs = usage_after.ru_nvcsw - usage_before.ru_nvcsw;
        long invol_cs = usage_after.ru_nivcsw - usage_before.ru_nivcsw;
        long minor_faults = usage_after.ru_minflt - usage_before.ru_minflt;
        long major_faults = usage_after.ru_majflt - usage_before.ru_majflt;

        double data_mb = (double)ny * nx * sizeof(float) / (1024.0 * 1024.0);
        double norm_mb = (double)ny * nx * sizeof(double) / (1024.0 * 1024.0);
        double result_mb = (double)ny * ny * sizeof(float) / (1024.0 * 1024.0);
        double total_mb = data_mb + norm_mb + result_mb;

        printf("%-25s | %-15s\n", "Metric", "Value");
        printf("--------------------------|----------------\n");
        printf("%-25s | %.2f ms\n", "Task Clock", task_clock_ms);
        printf("%-25s | %.2f\n", "CPUs Utilized", cpus_used);
        printf("%-25s | %ld\n", "Context Switches (vol)", vol_cs);
        printf("%-25s | %ld\n", "Context Switches (invol)", invol_cs);
        printf("%-25s | %ld\n", "Page Faults (minor)", minor_faults);
        printf("%-25s | %ld\n", "Page Faults (major)", major_faults);
        printf("%-25s | %.6f s\n", "Wall Clock Time", wall_time);
        printf("%-25s | %.2f MB\n", "Memory Footprint", total_mb);

        double max_err = 0.0;
        for (int i = 0; i < ny; i++) {
            double err = fabs(result[i + i * ny] - 1.0f);
            if (err > max_err) max_err = err;
        }
        printf("%-25s | %.2e\n", "Max Diagonal Error", max_err);

        free(data);
        free(result);
    }

    printf("\n");
    return 0;
}
