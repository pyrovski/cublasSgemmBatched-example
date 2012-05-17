all: gemm

gemm: gemm.cpp
	nvcc -o $@ $^ -arch=sm_20 -Xptxas -v -O3 -lcublas
