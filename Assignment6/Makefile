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

QA_BIN = questionA/device_query
QB_BIN = questionB/array_sum
QC_BIN = questionC/matrix_add

.PHONY: all clean run-a run-b run-c run-all

all: $(QA_BIN) $(QB_BIN) $(QC_BIN)

$(QA_BIN): questionA/device_query.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(QB_BIN): questionB/array_sum.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(QC_BIN): questionC/matrix_add.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-a: $(QA_BIN)
	./$(QA_BIN)

run-b: $(QB_BIN)
	./$(QB_BIN)

run-c: $(QC_BIN)
	./$(QC_BIN)

run-all: all
	./$(QA_BIN)
	./$(QB_BIN)
	./$(QC_BIN)

clean:
	rm -f $(QA_BIN) $(QB_BIN) $(QC_BIN)
