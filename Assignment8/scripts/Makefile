
CUDA12_LOCAL := ../../.local_cuda12/usr/local/cuda-12.6
VENV_NVIDIA_ROOT := ../../.venv/lib/python3.12/site-packages/nvidia
VENV_NVCC := $(VENV_NVIDIA_ROOT)/cu13/bin/nvcc
VENV_CUBLAS_INC := $(VENV_NVIDIA_ROOT)/cublas/include
VENV_CUBLAS_LIB := $(VENV_NVIDIA_ROOT)/cublas/lib
VENV_CUDNN_INC := $(VENV_NVIDIA_ROOT)/cudnn/include
VENV_CUDNN_LIB := $(VENV_NVIDIA_ROOT)/cudnn/lib

SM        := sm_75
NVCC      := nvcc
CUDA_HOME := $(CUDA12_LOCAL)

ifeq ($(wildcard $(CUDA12_LOCAL)/bin/nvcc), $(CUDA12_LOCAL)/bin/nvcc)
NVCC := $(CUDA12_LOCAL)/bin/nvcc
CUDA_HOME := $(CUDA12_LOCAL)
else ifeq ($(wildcard $(VENV_NVCC)), $(VENV_NVCC))
NVCC := $(VENV_NVCC)
CUDA_HOME := $(VENV_NVIDIA_ROOT)/cuda_runtime
endif

NVCCFLAGS := -O2 -arch=$(SM) -lineinfo \
	-I$(CUDA_HOME)/include \
	-I$(VENV_CUBLAS_INC) \
	-I$(VENV_CUDNN_INC) \
	-L$(CUDA_HOME)/lib64 \
	-L$(VENV_CUBLAS_LIB) \
	-L$(VENV_CUDNN_LIB) \
	-Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
	-Xlinker -rpath -Xlinker $(VENV_CUBLAS_LIB) \
	-Xlinker -rpath -Xlinker $(VENV_CUDNN_LIB) \
	-cudart shared

LIBS_BASE  := -lm
LIBS_BLAS  := -l:libcublas.so.12 $(LIBS_BASE)
LIBS_FULL  := -l:libcudnn.so.9 -l:libcublas.so.12 $(LIBS_BASE)

.PHONY: all clean ex01 ex02 ex03 ex04 ex05 \
        run_ex01 run_ex02 run_ex03 run_ex04

all: ex01 ex02 ex03 ex04 ex05

ex01: ex01_cuda_basics.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

ex02: ex02_memory_hierarchy.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

ex03: ex03_ml_primitives.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BASE)
	@echo "[✓] Built $@"

ex04: ex04_cnn_layers.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_BLAS)
	@echo "[✓] Built $@"

ex05: ex05_mnist_cnn.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS_FULL)
	@echo "[✓] Built $@"

run_ex01: ex01
	./ex01

run_ex02: ex02
	./ex02

run_ex03: ex03
	./ex03

run_ex04: ex04
	./ex04

clean:
	rm -f ex01 ex02 ex03 ex04 ex05
	@echo "[✓] Cleaned"
