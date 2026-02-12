#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define EPSILON 1.0
#define SIGMA   1.0
#define CUTOFF  2.5

void initialize_particles(double *x, double *y, double *z, int N, double L) {
    unsigned int seed = 42;
    for (int i = 0; i < N; i++) {
        x[i] = L * ((double)rand_r(&seed) / RAND_MAX);
        y[i] = L * ((double)rand_r(&seed) / RAND_MAX);
        z[i] = L * ((double)rand_r(&seed) / RAND_MAX);
    }
}

double compute_forces_sequential(double *x, double *y, double *z,
                                  double *fx, double *fy, double *fz,
                                  int N, double L) {
    double total_energy = 0.0;
    double cutoff2 = CUTOFF * CUTOFF;

    /* Zero out forces */
    for (int i = 0; i < N; i++) {
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }

    /* Nested loop over all unique pairs */
    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
            /* Minimum image convention for periodic boundaries */
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];

            /* Periodic boundary conditions */
            dx -= L * round(dx / L);
            dy -= L * round(dy / L);
            dz -= L * round(dz / L);

            double r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < cutoff2) {
                double r2inv = 1.0 / r2;
                double r6inv = r2inv * r2inv * r2inv;
                double sigma6 = SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA;

                /* Lennard-Jones potential: V = 4*eps*[(sig/r)^12 - (sig/r)^6] */
                double lj_energy = 4.0 * EPSILON * sigma6 * r6inv * (sigma6 * r6inv - 1.0);
                total_energy += lj_energy;

                /* Force: F = -dV/dr = 24*eps/r * [2*(sig/r)^12 - (sig/r)^6] */
                double force_mag = 24.0 * EPSILON * r2inv * sigma6 * r6inv * (2.0 * sigma6 * r6inv - 1.0);

                /* Newton's 3rd law: accumulate forces on both particles */
                fx[i] += force_mag * dx;
                fy[i] += force_mag * dy;
                fz[i] += force_mag * dz;
                fx[j] -= force_mag * dx;
                fy[j] -= force_mag * dy;
                fz[j] -= force_mag * dz;
            }
        }
    }

    return total_energy;
}

double compute_forces_parallel(double *x, double *y, double *z,
                                double *fx, double *fy, double *fz,
                                int N, double L, int num_threads) {
    double total_energy = 0.0;
    double cutoff2 = CUTOFF * CUTOFF;

    omp_set_num_threads(num_threads);

    /* Zero out forces */
    for (int i = 0; i < N; i++) {
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }

    /*
     * Strategy: Each thread maintains local force arrays to avoid race
     * conditions. After computation, local arrays are reduced into global.
     * Energy is reduced via OpenMP reduction clause.
     * Load balancing: schedule(dynamic) because inner loop size varies with i.
     */
    #pragma omp parallel reduction(+:total_energy)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /* Thread-local force arrays to avoid race conditions */
        double *lfx = (double *)calloc(N, sizeof(double));
        double *lfy = (double *)calloc(N, sizeof(double));
        double *lfz = (double *)calloc(N, sizeof(double));

        /* Dynamic scheduling for load balancing:
         * Outer loop iterations have decreasing work (N-1-i pairs).
         * Dynamic schedule distributes iterations to idle threads. */
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < N - 1; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                double dz = z[i] - z[j];

                /* Periodic boundary conditions */
                dx -= L * round(dx / L);
                dy -= L * round(dy / L);
                dz -= L * round(dz / L);

                double r2 = dx * dx + dy * dy + dz * dz;

                if (r2 < cutoff2) {
                    double r2inv = 1.0 / r2;
                    double r6inv = r2inv * r2inv * r2inv;
                    double sigma6 = SIGMA * SIGMA * SIGMA * SIGMA * SIGMA * SIGMA;

                    double lj_energy = 4.0 * EPSILON * sigma6 * r6inv * (sigma6 * r6inv - 1.0);
                    total_energy += lj_energy;

                    double force_mag = 24.0 * EPSILON * r2inv * sigma6 * r6inv * (2.0 * sigma6 * r6inv - 1.0);

                    /* Accumulate in local arrays — no race condition */
                    lfx[i] += force_mag * dx;
                    lfy[i] += force_mag * dy;
                    lfz[i] += force_mag * dz;
                    lfx[j] -= force_mag * dx;
                    lfy[j] -= force_mag * dy;
                    lfz[j] -= force_mag * dz;
                }
            }
        }

        /* Reduce local forces into global arrays (critical section) */
        #pragma omp critical
        {
            for (int i = 0; i < N; i++) {
                fx[i] += lfx[i];
                fy[i] += lfy[i];
                fz[i] += lfz[i];
            }
        }

        free(lfx);
        free(lfy);
        free(lfz);
    }

    return total_energy;
}

int main() {
    /* Particle counts to test */
    int sizes[] = {500, 1000, 2000, 4000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    /* Thread counts to test: 1 through 10 */
    int thread_counts[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    double L = 20.0;  /* Box length */

    /* Print hardware info — cores only */
    printf("Physical Cores: 4\n");
    printf("Logical Cores : 8 (Hyper-Threading)\n");
    printf("Max OpenMP threads: %d\n\n", omp_get_max_threads());

    printf("%-8s | %-8s | %-14s | %-14s | %-10s\n",
           "N", "Threads", "Seq Time (s)", "Par Time (s)", "Speedup");
    printf("---------|----------|----------------|----------------|----------\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];

        /* Allocate arrays */
        double *x  = (double *)malloc(N * sizeof(double));
        double *y  = (double *)malloc(N * sizeof(double));
        double *z  = (double *)malloc(N * sizeof(double));
        double *fx_seq = (double *)malloc(N * sizeof(double));
        double *fy_seq = (double *)malloc(N * sizeof(double));
        double *fz_seq = (double *)malloc(N * sizeof(double));
        double *fx_par = (double *)malloc(N * sizeof(double));
        double *fy_par = (double *)malloc(N * sizeof(double));
        double *fz_par = (double *)malloc(N * sizeof(double));

        initialize_particles(x, y, z, N, L);

        /* --- Sequential execution --- */
        double t1 = omp_get_wtime();
        double energy_seq = compute_forces_sequential(x, y, z, fx_seq, fy_seq, fz_seq, N, L);
        double t2 = omp_get_wtime();
        double seq_time = t2 - t1;

        /* --- Parallel execution with varying thread counts --- */
        for (int t = 0; t < num_thread_counts; t++) {
            int threads = thread_counts[t];

            double t3 = omp_get_wtime();
            double energy_par = compute_forces_parallel(x, y, z, fx_par, fy_par, fz_par, N, L, threads);
            double t4 = omp_get_wtime();
            double par_time = t4 - t3;

            double speedup = seq_time / par_time;

            printf("%-8d | %-8d | %-14.6f | %-14.6f | %-10.2f\n",
                   N, threads, seq_time, par_time, speedup);
        }

        printf("---------|----------|----------------|----------------|----------\n");

        free(x); free(y); free(z);
        free(fx_seq); free(fy_seq); free(fz_seq);
        free(fx_par); free(fy_par); free(fz_par);
    }

    /* === Detailed analysis for N=2000 with threads 1-10 === */
    printf("\n=== Detailed Analysis for N = 2000 ===\n\n");
    int N = 2000;
    double *x  = (double *)malloc(N * sizeof(double));
    double *y  = (double *)malloc(N * sizeof(double));
    double *z  = (double *)malloc(N * sizeof(double));
    double *fx = (double *)malloc(N * sizeof(double));
    double *fy = (double *)malloc(N * sizeof(double));
    double *fz = (double *)malloc(N * sizeof(double));

    initialize_particles(x, y, z, N, L);

    double t1 = omp_get_wtime();
    double energy = compute_forces_sequential(x, y, z, fx, fy, fz, N, L);
    double t2 = omp_get_wtime();
    double base_time = t2 - t1;

    printf("Total potential energy (N=%d): %.6f\n", N, energy);
    printf("Total pairs evaluated: %lld\n", (long long)N * (N - 1) / 2);
    printf("Sequential baseline time: %.6f s\n\n", base_time);

    printf("%-10s | %-14s | %-10s\n",
           "Threads", "Time (s)", "Speedup");
    printf("-----------|----------------|----------\n");

    for (int threads = 1; threads <= 10; threads++) {
        double t3 = omp_get_wtime();
        compute_forces_parallel(x, y, z, fx, fy, fz, N, L, threads);
        double t4 = omp_get_wtime();
        double par_time = t4 - t3;
        double speedup = base_time / par_time;

        printf("%-10d | %-14.6f | %-10.2f\n",
               threads, par_time, speedup);
    }

    free(x); free(y); free(z);
    free(fx); free(fy); free(fz);

    printf("\n");

    return 0;
}
