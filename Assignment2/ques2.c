#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <omp.h>

#define MATCH     2
#define MISMATCH -1
#define GAP      -1

void generate_sequence(char *seq, int len, unsigned int seed) {
    const char bases[] = "ACGT";
    for (int i = 0; i < len; i++) {
        seq[i] = bases[rand_r(&seed) % 4];
    }
    seq[len] = '\0';
}

static inline int max4(int a, int b, int c, int d) {
    int m1 = a > b ? a : b;
    int m2 = c > d ? c : d;
    return m1 > m2 ? m1 : m2;
}

static inline int match_score(char a, char b) {
    return (a == b) ? MATCH : MISMATCH;
}

int smith_waterman_seq(const char *seq1, int m, const char *seq2, int n, int *H) {
    int max_score = 0;
    int cols = n + 1;

    for (int i = 0; i <= m; i++) H[i * cols] = 0;
    for (int j = 0; j <= n; j++) H[j] = 0;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int val = max4(0,
                H[(i-1)*cols + (j-1)] + match_score(seq1[i-1], seq2[j-1]),
                H[(i-1)*cols + j] + GAP,
                H[i*cols + (j-1)] + GAP);
            H[i * cols + j] = val;
            if (val > max_score) max_score = val;
        }
    }
    return max_score;
}

int smith_waterman_wavefront(const char *seq1, int m, const char *seq2, int n,
                              int *H, int num_threads, int sched_type) {
    /* sched_type: 0=static, 1=dynamic, 2=guided */
    int max_score = 0;
    int cols = n + 1;

    omp_set_num_threads(num_threads);

    for (int i = 0; i <= m; i++) H[i * cols] = 0;
    for (int j = 0; j <= n; j++) H[j] = 0;

    /* Single parallel region - barrier between diagonals */
    #pragma omp parallel
    {
        int local_max = 0;

        for (int d = 2; d <= m + n; d++) {
            int i_start = (d - n > 1) ? d - n : 1;
            int i_end   = (d - 1 < m) ? d - 1 : m;

            if (sched_type == 0) {
                #pragma omp for schedule(static) nowait
                for (int i = i_start; i <= i_end; i++) {
                    int j = d - i;
                    int val = max4(0,
                        H[(i-1)*cols + (j-1)] + match_score(seq1[i-1], seq2[j-1]),
                        H[(i-1)*cols + j] + GAP,
                        H[i*cols + (j-1)] + GAP);
                    H[i * cols + j] = val;
                    if (val > local_max) local_max = val;
                }
            } else if (sched_type == 1) {
                #pragma omp for schedule(dynamic, 64) nowait
                for (int i = i_start; i <= i_end; i++) {
                    int j = d - i;
                    int val = max4(0,
                        H[(i-1)*cols + (j-1)] + match_score(seq1[i-1], seq2[j-1]),
                        H[(i-1)*cols + j] + GAP,
                        H[i*cols + (j-1)] + GAP);
                    H[i * cols + j] = val;
                    if (val > local_max) local_max = val;
                }
            } else {
                #pragma omp for schedule(guided, 32) nowait
                for (int i = i_start; i <= i_end; i++) {
                    int j = d - i;
                    int val = max4(0,
                        H[(i-1)*cols + (j-1)] + match_score(seq1[i-1], seq2[j-1]),
                        H[(i-1)*cols + j] + GAP,
                        H[i*cols + (j-1)] + GAP);
                    H[i * cols + j] = val;
                    if (val > local_max) local_max = val;
                }
            }

            /* Barrier: all cells on diagonal d must finish
               before any thread starts diagonal d+1 */
            #pragma omp barrier
        }

        #pragma omp critical
        {
            if (local_max > max_score) max_score = local_max;
        }
    }

    return max_score;
}

int main() {
    int sizes[] = {1000, 2000, 4000, 8000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Physical Cores: 4\n");
    printf("Logical Cores : 8 (Hyper-Threading)\n");
    printf("Max OpenMP threads: %d\n\n", omp_get_max_threads());

    /* ========== Part 1: Thread scaling (wavefront, static) ========== */
    printf("=== Part 1: Thread Scaling (Wavefront, Static Schedule) ===\n\n");
    printf("%-8s | %-8s | %-14s | %-14s | %-10s\n",
           "SeqLen", "Threads", "Seq Time (s)", "Par Time (s)", "Speedup");
    printf("---------|----------|----------------|----------------|----------\n");

    for (int s = 0; s < num_sizes; s++) {
        int len = sizes[s];
        char *seq1 = (char *)malloc((len + 1) * sizeof(char));
        char *seq2 = (char *)malloc((len + 1) * sizeof(char));
        int *H = (int *)calloc((long)(len + 1) * (len + 1), sizeof(int));

        generate_sequence(seq1, len, 12345);
        generate_sequence(seq2, len, 67890);

        /* Sequential */
        double t1 = omp_get_wtime();
        int score_seq = smith_waterman_seq(seq1, len, seq2, len, H);
        double t2 = omp_get_wtime();
        double seq_time = t2 - t1;

        /* Parallel with threads 1-10 */
        for (int threads = 1; threads <= 10; threads++) {
            double t3 = omp_get_wtime();
            int score_par = smith_waterman_wavefront(seq1, len, seq2, len, H, threads, 0);
            double t4 = omp_get_wtime();
            double par_time = t4 - t3;
            double speedup = seq_time / par_time;

            printf("%-8d | %-8d | %-14.6f | %-14.6f | %-10.2f\n",
                   len, threads, seq_time, par_time, speedup);

            if (score_seq != score_par) {
                printf("  WARNING: Score mismatch! Seq=%d Par=%d\n", score_seq, score_par);
            }
        }
        printf("---------|----------|----------------|----------------|----------\n");

        free(seq1); free(seq2); free(H);
    }

    /* ========== Part 2: Scheduling strategy comparison ========== */
    printf("\n=== Part 2: Scheduling Strategy Comparison (4 threads) ===\n\n");
    printf("%-8s | %-10s | %-14s | %-14s | %-10s\n",
           "SeqLen", "Schedule", "Seq Time (s)", "Par Time (s)", "Speedup");
    printf("---------|------------|----------------|----------------|----------\n");

    const char *sched_names[] = {"static", "dynamic", "guided"};

    for (int s = 0; s < num_sizes; s++) {
        int len = sizes[s];
        char *seq1 = (char *)malloc((len + 1) * sizeof(char));
        char *seq2 = (char *)malloc((len + 1) * sizeof(char));
        int *H = (int *)calloc((long)(len + 1) * (len + 1), sizeof(int));

        generate_sequence(seq1, len, 12345);
        generate_sequence(seq2, len, 67890);

        /* Sequential baseline */
        double t1 = omp_get_wtime();
        smith_waterman_seq(seq1, len, seq2, len, H);
        double t2 = omp_get_wtime();
        double seq_time = t2 - t1;

        for (int sc = 0; sc < 3; sc++) {
            double t3 = omp_get_wtime();
            smith_waterman_wavefront(seq1, len, seq2, len, H, 4, sc);
            double t4 = omp_get_wtime();
            double par_time = t4 - t3;
            double speedup = seq_time / par_time;

            printf("%-8d | %-10s | %-14.6f | %-14.6f | %-10.2f\n",
                   len, sched_names[sc], seq_time, par_time, speedup);
        }
        printf("---------|------------|----------------|----------------|----------\n");

        free(seq1); free(seq2); free(H);
    }

    /* ========== Part 3: Performance Stats (N=4000, 4 threads, static) ========== */
    printf("\n=== Part 3: Performance Stats (N=4000, 4 threads, static) ===\n\n");
    {
        int len = 4000;
        char *seq1 = (char *)malloc((len + 1) * sizeof(char));
        char *seq2 = (char *)malloc((len + 1) * sizeof(char));
        int *H = (int *)calloc((long)(len + 1) * (len + 1), sizeof(int));

        generate_sequence(seq1, len, 12345);
        generate_sequence(seq2, len, 67890);

        struct rusage usage_before, usage_after;
        getrusage(RUSAGE_SELF, &usage_before);

        clock_t cpu_start = clock();
        double wall_start = omp_get_wtime();

        smith_waterman_wavefront(seq1, len, seq2, len, H, 4, 0);

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

        printf("%-25s | %-15s\n", "Metric", "Value");
        printf("--------------------------|----------------\n");
        printf("%-25s | %.2f ms\n", "Task Clock", task_clock_ms);
        printf("%-25s | %.2f\n", "CPUs Utilized", cpus_used);
        printf("%-25s | %ld\n", "Context Switches (vol)", vol_cs);
        printf("%-25s | %ld\n", "Context Switches (invol)", invol_cs);
        printf("%-25s | %ld\n", "Page Faults (minor)", minor_faults);
        printf("%-25s | %ld\n", "Page Faults (major)", major_faults);
        printf("%-25s | %.6f s\n", "Wall Clock Time", wall_time);

        /* Scheduling policy summary */
        printf("\n=== Scheduling Policy Summary ===\n\n");
        printf("%-12s | %-28s | %-14s | %-25s\n",
               "Policy", "Work Distribution", "Overhead", "Observed Behaviour");
        printf("-------------|------------------------------|----------------|-------------------------\n");
        printf("%-12s | %-28s | %-14s | %-25s\n",
               "Static", "Assigned once at start", "Lowest", "Steady increase, best");
        printf("%-12s | %-28s | %-14s | %-25s\n",
               "Dynamic(64)", "Assigned at runtime in chunks", "High", "Slower than static");
        printf("%-12s | %-28s | %-14s | %-25s\n",
               "Guided(32)", "Adaptive chunk sizing", "Medium-High", "Unstable at high threads");

        free(seq1); free(seq2); free(H);
    }

    printf("\n");
    return 0;
}
