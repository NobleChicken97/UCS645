#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int is_perfect(int x) {
    if (x < 2) {
        return 0;
    }

    int sum = 1;
    for (int i = 2; i * i <= x; i++) {
        if (x % i == 0) {
            sum += i;
            int other = x / i;
            if (other != i) {
                sum += other;
            }
        }
    }
    return sum == x;
}

static int cmp_int(const void *a, const void *b) {
    const int ia = *(const int *)a;
    const int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int max_value = 10000;
    if (argc > 1) {
        max_value = atoi(argv[1]);
    }

    if (size < 2) {
        if (rank == 0) {
            int count = 0;
            printf("Q5 Perfect numbers up to %d (single process fallback)\n", max_value);
            for (int n = 2; n <= max_value; n++) {
                if (is_perfect(n)) {
                    printf("%d ", n);
                    count++;
                }
            }
            printf("\nTotal perfect numbers: %d\n", count);
        }
        MPI_Finalize();
        return 0;
    }

    const int stop_signal = 0;

    if (rank == 0) {
        int capacity = max_value > 0 ? max_value : 1;
        int *perfects = (int *)malloc((size_t)capacity * sizeof(int));
        if (!perfects) {
            fprintf(stderr, "Master allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int count = 0;
        int next = 2;
        int active_workers = size - 1;

        while (active_workers > 0) {
            int response = 0;
            MPI_Status status;
            MPI_Recv(&response, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            if (response > 0) {
                perfects[count++] = response;
            }

            int worker = status.MPI_SOURCE;
            if (next <= max_value) {
                MPI_Send(&next, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
                next++;
            } else {
                MPI_Send(&stop_signal, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        qsort(perfects, (size_t)count, sizeof(int), cmp_int);

        printf("Q5 Perfect numbers up to %d\n", max_value);
        for (int i = 0; i < count; i++) {
            printf("%d ", perfects[i]);
        }
        printf("\nTotal perfect numbers: %d\n", count);

        free(perfects);
    } else {
        int response = 0;
        MPI_Send(&response, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        while (1) {
            int task = 0;
            MPI_Recv(&task, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (task == stop_signal) {
                break;
            }

            response = is_perfect(task) ? task : -task;
            MPI_Send(&response, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
