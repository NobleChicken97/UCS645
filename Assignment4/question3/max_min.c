#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int nums[10];
    int local_max, local_min;
    int global_max, global_min;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(42 + rank);
    for (int i = 0; i < 10; i++)
        nums[i] = rand() % 1001;

    local_max = nums[0];
    local_min = nums[0];
    for (int i = 1; i < 10; i++) {
        if (nums[i] > local_max) local_max = nums[i];
        if (nums[i] < local_min) local_min = nums[i];
    }

    printf("Process %d: local_max = %d, local_min = %d\n", rank, local_max, local_min);

    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    struct { int val; int rank; } local_max_loc, global_max_loc, local_min_loc, global_min_loc;

    local_max_loc.val = local_max;
    local_max_loc.rank = rank;
    local_min_loc.val = local_min;
    local_min_loc.rank = rank;

    MPI_Reduce(&local_max_loc, &global_max_loc, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min_loc, &global_min_loc, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global Max = %d (Process %d)\n", global_max, global_max_loc.rank);
        printf("Global Min = %d (Process %d)\n", global_min, global_min_loc.rank);
    }

    MPI_Finalize();
    return 0;
}
