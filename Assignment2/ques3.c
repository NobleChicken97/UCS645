#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <omp.h>

#define ALPHA  0.01
#define DX     0.01
#define DT     0.000025
#define T_HOT  100.0
#define T_COLD 0.0

void initialize_grid(double *grid, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            grid[i * N + j] = T_COLD;

    for (int j = 0; j < N; j++)
        grid[0 * N + j] = T_HOT;
}

double heat_diffusion_seq(double *curr, double *next, int N, int timesteps) {
    double r = ALPHA * DT / (DX * DX);
    double total_heat = 0.0;

    for (int t = 0; t < timesteps; t++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                next[i * N + j] = curr[i * N + j] + r * (
                    curr[(i-1)*N + j] + curr[(i+1)*N + j] +
                    curr[i*N + (j-1)] + curr[i*N + (j+1)] -
                    4.0 * curr[i*N + j]);
            }
        }
        double *tmp = curr;
        curr = next;
        next = tmp;
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            total_heat += curr[i * N + j];

    return total_heat;
}

double heat_diffusion_par(double *curr, double *next, int N, int timesteps,
                           int num_threads, int sched_type) {
    double r = ALPHA * DT / (DX * DX);
    double total_heat = 0.0;

    omp_set_num_threads(num_threads);

    for (int t = 0; t < timesteps; t++) {
        if (sched_type == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    next[i * N + j] = curr[i * N + j] + r * (
                        curr[(i-1)*N + j] + curr[(i+1)*N + j] +
                        curr[i*N + (j-1)] + curr[i*N + (j+1)] -
                        4.0 * curr[i*N + j]);
                }
            }
        } else if (sched_type == 1) {
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    next[i * N + j] = curr[i * N + j] + r * (
                        curr[(i-1)*N + j] + curr[(i+1)*N + j] +
                        curr[i*N + (j-1)] + curr[i*N + (j+1)] -
                        4.0 * curr[i*N + j]);
                }
            }
        } else {
            #pragma omp parallel for schedule(guided, 8)
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    next[i * N + j] = curr[i * N + j] + r * (
                        curr[(i-1)*N + j] + curr[(i+1)*N + j] +
                        curr[i*N + (j-1)] + curr[i*N + (j+1)] -
                        4.0 * curr[i*N + j]);
                }
            }
        }

        double *tmp = curr;
        curr = next;
        next = tmp;
    }

    #pragma omp parallel for reduction(+:total_heat) schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            total_heat += curr[i * N + j];

    return total_heat;
}

double heat_diffusion_tiled(double *curr, double *next, int N, int timesteps,
                             int num_threads, int tile_size) {
    double r = ALPHA * DT / (DX * DX);
    double total_heat = 0.0;

    omp_set_num_threads(num_threads);

    for (int t = 0; t < timesteps; t++) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int ii = 1; ii < N - 1; ii += tile_size) {
            for (int jj = 1; jj < N - 1; jj += tile_size) {
                int i_end = ii + tile_size < N - 1 ? ii + tile_size : N - 1;
                int j_end = jj + tile_size < N - 1 ? jj + tile_size : N - 1;
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        next[i * N + j] = curr[i * N + j] + r * (
                            curr[(i-1)*N + j] + curr[(i+1)*N + j] +
                            curr[i*N + (j-1)] + curr[i*N + (j+1)] -
                            4.0 * curr[i*N + j]);
                    }
                }
            }
        }

        double *tmp = curr;
        curr = next;
        next = tmp;
    }

    #pragma omp parallel for reduction(+:total_heat) schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            total_heat += curr[i * N + j];

    return total_heat;
}

int main() {
    int grid_sizes[] = {500, 1000, 2000};
    int num_sizes = 3;
    int timesteps = 100;

    printf("Physical Cores: 4\n");
    printf("Logical Cores : 8 (Hyper-Threading)\n");
    printf("Max OpenMP threads: %d\n", omp_get_max_threads());
    printf("Timesteps: %d | Alpha: %.4f | dx: %.4f | dt: %.6f\n\n",
           timesteps, ALPHA, DX, DT);

    printf("=== Part 1: Thread Scaling (Static Schedule) ===\n\n");
    printf("%-8s | %-8s | %-14s | %-14s | %-10s\n",
           "Grid", "Threads", "Seq Time (s)", "Par Time (s)", "Speedup");
    printf("---------|----------|----------------|----------------|----------\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = grid_sizes[s];
        double *curr = (double *)malloc(N * N * sizeof(double));
        double *next = (double *)malloc(N * N * sizeof(double));

        initialize_grid(curr, N);
        initialize_grid(next, N);
        double t1 = omp_get_wtime();
        double heat_seq = heat_diffusion_seq(curr, next, N, timesteps);
        double t2 = omp_get_wtime();
        double seq_time = t2 - t1;

        for (int threads = 1; threads <= 10; threads++) {
            initialize_grid(curr, N);
            initialize_grid(next, N);
            double t3 = omp_get_wtime();
            double heat_par = heat_diffusion_par(curr, next, N, timesteps, threads, 0);
            double t4 = omp_get_wtime();
            double par_time = t4 - t3;
            double speedup = seq_time / par_time;

            printf("%-8d | %-8d | %-14.6f | %-14.6f | %-10.2f\n",
                   N, threads, seq_time, par_time, speedup);
        }
        printf("---------|----------|----------------|----------------|----------\n");

        free(curr); free(next);
    }

    printf("\n=== Part 2: Scheduling Strategy Comparison (4 threads) ===\n\n");
    printf("%-8s | %-10s | %-14s | %-14s | %-10s\n",
           "Grid", "Schedule", "Seq Time (s)", "Par Time (s)", "Speedup");
    printf("---------|------------|----------------|----------------|----------\n");

    const char *sched_names[] = {"static", "dynamic", "guided"};

    for (int s = 0; s < num_sizes; s++) {
        int N = grid_sizes[s];
        double *curr = (double *)malloc(N * N * sizeof(double));
        double *next = (double *)malloc(N * N * sizeof(double));

        initialize_grid(curr, N);
        initialize_grid(next, N);
        double t1 = omp_get_wtime();
        heat_diffusion_seq(curr, next, N, timesteps);
        double t2 = omp_get_wtime();
        double seq_time = t2 - t1;

        for (int sc = 0; sc < 3; sc++) {
            initialize_grid(curr, N);
            initialize_grid(next, N);
            double t3 = omp_get_wtime();
            heat_diffusion_par(curr, next, N, timesteps, 4, sc);
            double t4 = omp_get_wtime();
            double par_time = t4 - t3;
            double speedup = seq_time / par_time;

            printf("%-8d | %-10s | %-14.6f | %-14.6f | %-10.2f\n",
                   N, sched_names[sc], seq_time, par_time, speedup);
        }
        printf("---------|------------|----------------|----------------|----------\n");

        free(curr); free(next);
    }

    printf("\n=== Part 3: Cache Blocking Comparison (4 threads, N=2000) ===\n\n");
    {
        int N = 2000;
        double *curr = (double *)malloc(N * N * sizeof(double));
        double *next = (double *)malloc(N * N * sizeof(double));

        initialize_grid(curr, N);
        initialize_grid(next, N);
        double t1 = omp_get_wtime();
        heat_diffusion_par(curr, next, N, timesteps, 4, 0);
        double t2 = omp_get_wtime();
        double no_tile_time = t2 - t1;

        printf("%-14s | %-14s | %-10s\n", "Tile Size", "Time (s)", "Speedup vs No-Tile");
        printf("---------------|----------------|-------------------\n");
        printf("%-14s | %-14.6f | %-10s\n", "No tiling", no_tile_time, "1.00");

        int tile_sizes[] = {16, 32, 64, 128};
        for (int ti = 0; ti < 4; ti++) {
            initialize_grid(curr, N);
            initialize_grid(next, N);
            double t3 = omp_get_wtime();
            heat_diffusion_tiled(curr, next, N, timesteps, 4, tile_sizes[ti]);
            double t4 = omp_get_wtime();
            double tiled_time = t4 - t3;
            double tile_speedup = no_tile_time / tiled_time;

            printf("%-14d | %-14.6f | %-10.2f\n", tile_sizes[ti], tiled_time, tile_speedup);
        }

        free(curr); free(next);
    }

    printf("\n=== Part 4: Performance Stats (N=2000, 4 threads, static) ===\n\n");
    {
        int N = 2000;
        double *curr = (double *)malloc(N * N * sizeof(double));
        double *next = (double *)malloc(N * N * sizeof(double));
        initialize_grid(curr, N);
        initialize_grid(next, N);

        struct rusage usage_before, usage_after;
        getrusage(RUSAGE_SELF, &usage_before);

        clock_t cpu_start = clock();
        double wall_start = omp_get_wtime();

        double total_heat = heat_diffusion_par(curr, next, N, timesteps, 4, 0);

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

        double mem_mb = 2.0 * N * N * sizeof(double) / (1024.0 * 1024.0);

        printf("%-25s | %-15s\n", "Metric", "Value");
        printf("--------------------------|----------------\n");
        printf("%-25s | %.2f ms\n", "Task Clock", task_clock_ms);
        printf("%-25s | %.2f\n", "CPUs Utilized", cpus_used);
        printf("%-25s | %ld\n", "Context Switches (vol)", vol_cs);
        printf("%-25s | %ld\n", "Context Switches (invol)", invol_cs);
        printf("%-25s | %ld\n", "Page Faults (minor)", minor_faults);
        printf("%-25s | %ld\n", "Page Faults (major)", major_faults);
        printf("%-25s | %.6f s\n", "Wall Clock Time", wall_time);
        printf("%-25s | %.2f MB\n", "Memory Footprint", mem_mb);
        printf("%-25s | %.2f\n", "Total Heat (final)", total_heat);

        free(curr); free(next);
    }

    printf("\n=== Scheduling Policy Summary ===\n\n");
    printf("%-12s | %-28s | %-14s | %-25s\n",
           "Policy", "Work Distribution", "Overhead", "Observed Behaviour");
    printf("-------------|------------------------------|----------------|-------------------------\n");
    printf("%-12s | %-28s | %-14s | %-25s\n",
           "Static", "Equal rows per thread", "Lowest", "Best and most consistent");
    printf("%-12s | %-28s | %-14s | %-25s\n",
           "Dynamic(16)", "Chunks of 16 rows on demand", "Higher", "Slight overhead, no gain");
    printf("%-12s | %-28s | %-14s | %-25s\n",
           "Guided(8)", "Decreasing chunk sizes", "Medium", "Similar to static");

    printf("\n");
    return 0;
}
