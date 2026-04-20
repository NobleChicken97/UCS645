CUDA12_LOCAL = ../.local_cuda12/usr/local/cuda-12.6
VENV_NVIDIA_ROOT = ../.venv/lib/python3.12/site-packages/nvidia
VENV_NVCC = $(VENV_NVIDIA_ROOT)/cu13/bin/nvcc

NVCC ?= nvcc
CUDA_HOME ?= $(CUDA12_LOCAL)

ifeq ($(wildcard $(CUDA12_LOCAL)/bin/nvcc), $(CUDA12_LOCAL)/bin/nvcc)
NVCC := $(CUDA12_LOCAL)/bin/nvcc
CUDA_HOME := $(CUDA12_LOCAL)
else ifeq ($(wildcard $(VENV_NVCC)), $(VENV_NVCC))
NVCC := $(VENV_NVCC)
CUDA_HOME := $(VENV_NVIDIA_ROOT)/cuda_runtime
endif

NVCCFLAGS = -O2 \
	-arch=sm_75 \
	-I$(CUDA_HOME)/include \
	-L$(CUDA_HOME)/lib64 \
	-Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
	-cudart shared

Q1_BIN = question1/problem1_threads_tasks
Q2_BIN = question2/problem2_merge_sort
Q3_BIN = question3/problem3_vector_add_profile

.PHONY: all clean run-1 run-2 run-3 run-all

all: $(Q1_BIN) $(Q2_BIN) $(Q3_BIN)

$(Q1_BIN): question1/problem1_threads_tasks.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(Q2_BIN): question2/problem2_merge_sort.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -std=c++17 -o $@ $<

$(Q3_BIN): question3/problem3_vector_add_profile.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-1: $(Q1_BIN)
	./$(Q1_BIN)

run-2: $(Q2_BIN)
	./$(Q2_BIN)

run-3: $(Q3_BIN)
	./$(Q3_BIN)

run-all: all
	./$(Q1_BIN)
	./$(Q2_BIN)
	./$(Q3_BIN)

clean:
	rm -f $(Q1_BIN) $(Q2_BIN) $(Q3_BIN)
