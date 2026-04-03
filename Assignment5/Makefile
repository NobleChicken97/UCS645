CC = mpicc
CFLAGS = -O2 -Wall -Wextra -std=c11

Q1_BIN = question1/daxpy_mpi
Q2_BIN = question2/broadcast_race
Q3_BIN = question3/distributed_dot
Q4_BIN = question4/prime_master_worker
Q5_BIN = question5/perfect_master_worker

.PHONY: all clean run-q1 run-q2 run-q3 run-q4 run-q5

all: $(Q1_BIN) $(Q2_BIN) $(Q3_BIN) $(Q4_BIN) $(Q5_BIN)

$(Q1_BIN): question1/daxpy_mpi.c
	$(CC) $(CFLAGS) -o $@ $<

$(Q2_BIN): question2/broadcast_race.c
	$(CC) $(CFLAGS) -o $@ $<

$(Q3_BIN): question3/distributed_dot.c
	$(CC) $(CFLAGS) -o $@ $<

$(Q4_BIN): question4/prime_master_worker.c
	$(CC) $(CFLAGS) -o $@ $<

$(Q5_BIN): question5/perfect_master_worker.c
	$(CC) $(CFLAGS) -o $@ $<

run-q1: $(Q1_BIN)
	mpirun --oversubscribe -np 4 ./$(Q1_BIN)

run-q2: $(Q2_BIN)
	mpirun --oversubscribe -np 4 ./$(Q2_BIN)

run-q3: $(Q3_BIN)
	mpirun --oversubscribe -np 4 ./$(Q3_BIN)

run-q4: $(Q4_BIN)
	mpirun --oversubscribe -np 4 ./$(Q4_BIN)

run-q5: $(Q5_BIN)
	mpirun --oversubscribe -np 4 ./$(Q5_BIN)

clean:
	rm -f $(Q1_BIN) $(Q2_BIN) $(Q3_BIN) $(Q4_BIN) $(Q5_BIN)
