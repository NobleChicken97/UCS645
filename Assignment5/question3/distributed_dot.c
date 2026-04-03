#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static double vector_x(long long i) {
    return 1.0 + (double)(i % 1000) * 0.001;
}

static double vector_y(long long i) {
    return 0.5 + (double)((i * 3) % 1000) * 0.002;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long n = 100000000LL;
    double multiplier = 2.0;

    if (argc > 1) {
        n = atoll(argv[1]);
    }
    if (argc > 2) {
        multiplier = atof(argv[2]);
    }

    if (rank == 0) {
        printf("Q3 Distributed Dot Product (N=%lld, np=%d)\n", n, size);
    }

    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    long long base = n / size;
    long long rem = n % size;
    long long local_n = base + (rank < rem ? 1 : 0);
    long long start = rank * base + (rank < rem ? rank : rem);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double local_dot = 0.0;
    for (long long i = 0; i < local_n; i++) {
        long long idx = start + i;
        double x = multiplier * vector_x(idx);
        double y = vector_y(idx);
        local_dot += x * y;
    }

    double global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;

    if (rank == 0) {
        printf("Multiplier: %.6f\n", multiplier);
        printf("Dot product: %.12f\n", global_dot);
        printf("MPI time: %.6f s\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}
