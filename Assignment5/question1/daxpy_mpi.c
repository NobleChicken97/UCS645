#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_vectors(double *x, double *y, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = 1.0 + (double)(i % 1000) * 0.001;
        y[i] = 2.0 + (double)(i % 2000) * 0.0005;
    }
}

static double checksum(const double *arr, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1 << 16;
    double a = 2.5;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        a = atof(argv[2]);
    }

    int chunk = (n + size - 1) / size;
    int padded_n = chunk * size;

    double *x = NULL;
    double *y = NULL;
    double *x_seq = NULL;
    double seq_time = 0.0;

    if (rank == 0) {
        x = (double *)malloc((size_t)padded_n * sizeof(double));
        y = (double *)malloc((size_t)padded_n * sizeof(double));
        x_seq = (double *)malloc((size_t)n * sizeof(double));
        if (!x || !y || !x_seq) {
            fprintf(stderr, "Rank 0 allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fill_vectors(x, y, n);
        for (int i = n; i < padded_n; i++) {
            x[i] = 0.0;
            y[i] = 0.0;
        }

        for (int i = 0; i < n; i++) {
            x_seq[i] = x[i];
        }

        double t0 = MPI_Wtime();
        for (int i = 0; i < n; i++) {
            x_seq[i] = a * x_seq[i] + y[i];
        }
        seq_time = MPI_Wtime() - t0;
    }

    double *x_local = (double *)malloc((size_t)chunk * sizeof(double));
    double *y_local = (double *)malloc((size_t)chunk * sizeof(double));
    if (!x_local || !y_local) {
        fprintf(stderr, "Rank %d allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    MPI_Scatter(x, chunk, MPI_DOUBLE, x_local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, chunk, MPI_DOUBLE, y_local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++) {
        x_local[i] = a * x_local[i] + y_local[i];
    }

    MPI_Gather(x_local, chunk, MPI_DOUBLE, x, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    double mpi_time = MPI_Wtime() - t1;

    if (rank == 0) {
        double parallel_checksum = checksum(x, n);
        double sequential_checksum = checksum(x_seq, n);
        double speedup = (mpi_time > 0.0) ? (seq_time / mpi_time) : 0.0;

        printf("Q1 DAXPY (N=%d, a=%.3f)\n", n, a);
        printf("Sequential time : %.6f s\n", seq_time);
        printf("MPI time        : %.6f s\n", mpi_time);
        printf("Speedup         : %.6f\n", speedup);
        printf("Sequential checksum: %.12f\n", sequential_checksum);
        printf("Parallel checksum  : %.12f\n", parallel_checksum);
        printf("Absolute error     : %.12e\n", fabs(sequential_checksum - parallel_checksum));
    }

    free(x_local);
    free(y_local);
    if (rank == 0) {
        free(x);
        free(y);
        free(x_seq);
    }

    MPI_Finalize();
    return 0;
}
