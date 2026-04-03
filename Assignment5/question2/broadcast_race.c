#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0

static void my_bcast(double *data, int count, int root, MPI_Comm comm) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root) {
        for (int dest = 0; dest < size; dest++) {
            if (dest != root) {
                MPI_Send(data, count, MPI_DOUBLE, dest, 0, comm);
            }
        }
    } else {
        MPI_Recv(data, count, MPI_DOUBLE, root, 0, comm, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10000000;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    double *arr = (double *)malloc((size_t)n * sizeof(double));
    if (!arr) {
        fprintf(stderr, "Rank %d failed to allocate %.2f MB\n", rank,
                (double)n * sizeof(double) / (1024.0 * 1024.0));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == ROOT) {
        for (int i = 0; i < n; i++) {
            arr[i] = (double)i * 0.001;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    my_bcast(arr, n, ROOT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double mybcast_time = MPI_Wtime() - t0;

    if (rank == ROOT) {
        for (int i = 0; i < n; i++) {
            arr[i] = (double)i * 0.002;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    MPI_Bcast(arr, n, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double mpi_bcast_time = MPI_Wtime() - t0;

    if (rank == ROOT) {
        double ratio = (mpi_bcast_time > 0.0) ? (mybcast_time / mpi_bcast_time) : 0.0;
        printf("Q2 Broadcast Race (N=%d doubles, np=%d)\n", n, size);
        printf("MyBcast time   : %.6f s\n", mybcast_time);
        printf("MPI_Bcast time : %.6f s\n", mpi_bcast_time);
        printf("MyBcast/MPI_Bcast: %.6f\n", ratio);
    }

    free(arr);
    MPI_Finalize();
    return 0;
}
