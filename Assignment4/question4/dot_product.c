#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 8;
    int A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int B[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    int *send_A = NULL;
    int *send_B = NULL;
    int local_dot = 0, global_dot = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = (N + size - 1) / size;
    int padded_N = chunk_size * size;

    int *local_A = (int *)malloc(chunk_size * sizeof(int));
    int *local_B = (int *)malloc(chunk_size * sizeof(int));

    if (local_A == NULL || local_B == NULL) {
        fprintf(stderr, "Process %d: memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        send_A = (int *)malloc(padded_N * sizeof(int));
        send_B = (int *)malloc(padded_N * sizeof(int));

        if (send_A == NULL || send_B == NULL) {
            fprintf(stderr, "Process 0: memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N; i++) {
            send_A[i] = A[i];
            send_B[i] = B[i];
        }
        for (int i = N; i < padded_N; i++) {
            send_A[i] = 0;
            send_B[i] = 0;
        }
    }

    MPI_Scatter(send_A, chunk_size, MPI_INT, local_A, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(send_B, chunk_size, MPI_INT, local_B, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk_size; i++)
        local_dot += local_A[i] * local_B[i];

    printf("Process %d: local_dot = %d\n", rank, local_dot);

    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Global Dot Product = %d\n", global_dot);

    if (rank == 0) {
        free(send_A);
        free(send_B);
    }

    free(local_A);
    free(local_B);
    MPI_Finalize();
    return 0;
}
