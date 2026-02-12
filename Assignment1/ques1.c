#include <stdio.h>
#include <omp.h>

int main() {
    int N = 65536;
    double X[N], Y[N];
    double a = 2.5;
    int i, threads;
    double t1, t2;
    double seq_time;

    for (i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    for (i = 0; i < N; i++)
        X[i] = 1.0;

    t1 = omp_get_wtime();
    for (i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }
    t2 = omp_get_wtime();
    seq_time = t2 - t1;

    printf("Sequential Time = %f seconds\n\n", seq_time);

    for (threads = 2; threads <= 12; threads++) {
        double par_time, speedup;

        omp_set_num_threads(threads);

        for (i = 0; i < N; i++)
            X[i] = 1.0;

        t1 = omp_get_wtime();
        #pragma omp parallel for
        for (i = 0; i < N; i++) {
            X[i] = a * X[i] + Y[i];
        }
        t2 = omp_get_wtime();

        par_time = t2 - t1;
        speedup = seq_time / par_time;

        printf("Threads = %d   Time = %f   Speedup = %.2f\n", threads, par_time, speedup);
    }

    return 0;
}
