#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int is_prime(int x) {
    if (x < 2) {
        return 0;
    }
    if (x == 2) {
        return 1;
    }
    if (x % 2 == 0) {
        return 0;
    }
    for (int i = 3; i * i <= x; i += 2) {
        if (x % i == 0) {
            return 0;
        }
    }
    return 1;
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

    int max_value = 200;
    if (argc > 1) {
        max_value = atoi(argv[1]);
    }

    if (size < 2) {
        if (rank == 0) {
            int count = 0;
            printf("Q4 Primes up to %d (single process fallback)\n", max_value);
            for (int n = 2; n <= max_value; n++) {
                if (is_prime(n)) {
                    printf("%d ", n);
                    count++;
                }
            }
            printf("\nTotal primes: %d\n", count);
        }
        MPI_Finalize();
        return 0;
    }

    const int stop_signal = 0;

    if (rank == 0) {
        int capacity = max_value > 0 ? max_value : 1;
        int *primes = (int *)malloc((size_t)capacity * sizeof(int));
        if (!primes) {
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
                primes[count++] = response;
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

        qsort(primes, (size_t)count, sizeof(int), cmp_int);

        printf("Q4 Primes up to %d\n", max_value);
        for (int i = 0; i < count; i++) {
            printf("%d ", primes[i]);
            if ((i + 1) % 20 == 0) {
                printf("\n");
            }
        }
        printf("\nTotal primes: %d\n", count);

        free(primes);
    } else {
        int response = 0;
        MPI_Send(&response, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        while (1) {
            int task = 0;
            MPI_Recv(&task, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (task == stop_signal) {
                break;
            }

            response = is_prime(task) ? task : -task;
            MPI_Send(&response, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
