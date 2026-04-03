#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 100;
    int *array = NULL;
    int chunk_size;
    int local_sum = 0, global_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    chunk_size = (N + size - 1) / size;
    int padded_N = chunk_size * size;
    int *local_array = (int *)malloc(chunk_size * sizeof(int));

    if (local_array == NULL) {
        fprintf(stderr, "Process %d: memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        array = (int *)malloc(padded_N * sizeof(int));
        if (array == NULL) {
            fprintf(stderr, "Process 0: memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        for (int i = N; i < padded_N; i++)
            array[i] = 0;
    }

    MPI_Scatter(array, chunk_size, MPI_INT, local_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk_size; i++)
        local_sum += local_array[i];

    printf("Process %d: local_sum = %d\n", rank, local_sum);

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global Sum = %d\n", global_sum);
        printf("Average = %.2f\n", (double)global_sum / N);
        free(array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
