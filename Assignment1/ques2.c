#include <stdio.h>
#include <omp.h>

double A[1000][1000], B[1000][1000], C[1000][1000];

int main()
{
    int N = 1000;
    int i, j, k, threads;
    double t1, t2, seq_time;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
            C[i][j] = 0.0;
        }
    }

    // sequential
    printf("Sequential:\n");
    t1 = omp_get_wtime();
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    t2 = omp_get_wtime();
    seq_time = t2 - t1;
    printf("Time = %f\n\n", seq_time);

    // 1D parallel
    printf("1D Parallel:\n");
    for (threads = 2; threads <= 10; threads++)
    {
        double par_time, speedup;

        omp_set_num_threads(threads);

        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                C[i][j] = 0.0;

        t1 = omp_get_wtime();
#pragma omp parallel for private(j, k)
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                for (k = 0; k < N; k++)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        t2 = omp_get_wtime();

        par_time = t2 - t1;
        speedup = seq_time / par_time;

        printf("Threads=%d  Time=%f  Speedup=%.2f\n", threads, par_time, speedup);
    }

    // 2D parallel
    printf("\n2D Parallel:\n");
    for (threads = 2; threads <= 10; threads++)
    {
        double par_time, speedup;

        omp_set_num_threads(threads);

        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                C[i][j] = 0.0;

        t1 = omp_get_wtime();
#pragma omp parallel for collapse(2) private(k)
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                for (k = 0; k < N; k++)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        t2 = omp_get_wtime();

        par_time = t2 - t1;
        speedup = seq_time / par_time;

        printf("Threads=%d  Time=%f  Speedup=%.2f\n", threads, par_time, speedup);
    }

    return 0;
}
