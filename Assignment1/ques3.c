#include <stdio.h>
#include <omp.h>

int main() {
    long num_steps = 100000;
    double step;
    int i, threads;
    double x, pi, sum;
    double t1, t2, seq_time;

    step = 1.0 / (double)num_steps;

    // sequential
    sum = 0.0;
    t1 = omp_get_wtime();
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    t2 = omp_get_wtime();
    seq_time = t2 - t1;

    printf("Sequential:\n");
    printf("Pi = %.10f\n", pi);
    printf("Time = %f\n\n", seq_time);

    // parallel
    printf("Parallel:\n");
    for (threads = 2; threads <= 10; threads++) {
        double par_time, speedup;

        omp_set_num_threads(threads);

        sum = 0.0;
        t1 = omp_get_wtime();
        #pragma omp parallel for private(x) reduction(+:sum)
        for (i = 0; i < num_steps; i++) {
            x = (i + 0.5) * step;
            sum = sum + 4.0 / (1.0 + x * x);
        }
        pi = step * sum;
        t2 = omp_get_wtime();

        par_time = t2 - t1;
        speedup = seq_time / par_time;

        printf("Threads=%d  Pi=%.10f  Time=%f  Speedup=%.2f\n", threads, pi, par_time, speedup);
    }

    return 0;
}
