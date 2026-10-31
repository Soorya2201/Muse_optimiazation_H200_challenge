CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

TK_DIR    := third-party/ThunderKittens

NVCC_FLAGS := \
	-std=c++20 \
	-gencode=arch=compute_90a,code=sm_90a \
	--expt-relaxed-constexpr \
	--extended-lambda \
	-lineinfo \
	-Xcompiler -fPIC \
	-DKITTENS_HOPPER \
	-I src \
	-I $(TK_DIR)/include \
	-I $(TK_DIR)/prototype

BUILD_DIR := build
SRC       := src/unweight.cu
OBJ       := $(BUILD_DIR)/unweight.o
LIB       := $(BUILD_DIR)/libunweight.a

.PHONY: all lib clean

all: lib

lib: $(LIB)

$(LIB): $(OBJ)
	ar rcs $@ $^

$(OBJ): $(SRC) $(wildcard src/*.cuh) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
