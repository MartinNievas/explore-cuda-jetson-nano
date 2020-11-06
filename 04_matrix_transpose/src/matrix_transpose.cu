#include<stdio.h>
#include<omp.h>
#include "helper_cuda.h"


#define N 2048l
#define BLOCK_SIZE 32l

__global__
void matrix_initialization(int * const __restrict__ matrix, size_t size) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  matrix[index] = index;

}

__global__
void matrix_transpose_naive(int * const __restrict__ input, int * const __restrict__ output) {

  int indexX = threadIdx.x + blockIdx.x * blockDim.x;
  int indexY = threadIdx.y + blockIdx.y * blockDim.y;
  int index = indexY * N + indexX;
  int transposedIndex = indexX * N + indexY;

  // index increase by 1, but transpoIndex by N
  output[transposedIndex] = input[index];

  // index increase by 1, but transpoIndex by N
  // output[index] = input[transposedIndex];
}

__global__
void matrix_transpose_shared(int * const __restrict__ input, int * const __restrict__ output) {

  __shared__ int sharedMemory [BLOCK_SIZE] [BLOCK_SIZE];

  int indexX = threadIdx.x + blockIdx.x * blockDim.x;
  int indexY = threadIdx.y + blockIdx.y * blockDim.y;

  int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
  int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

  int localIndexX = threadIdx.x;
  int localIndexY = threadIdx.y;

  int index = indexY * N + indexX;
  int transposedIndex = tindexY * N + tindexX;

  sharedMemory[localIndexX][localIndexY] = input[index];

  __syncthreads();

  output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
}

__global__
void matrix_transpose_shared_non_conflict(int * const __restrict__ input, int * const __restrict__ output) {

  __shared__ int sharedMemory [BLOCK_SIZE] [BLOCK_SIZE+1];
  // [1 ] [2 ] [3 ] ... [29] [30] [31] [32]

  // now 1 && 33 non bank conflict
  // [33] [34] [35] ... [61] [62] [63]

  // before 1&&32 bank conflict
  // [32] [33] [34] ... [60] [61] [62]

  int indexX = threadIdx.x + blockIdx.x * blockDim.x;
  int indexY = threadIdx.y + blockIdx.y * blockDim.y;

  int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
  int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

  int localIndexX = threadIdx.x;
  int localIndexY = threadIdx.y;

  int index = indexY * N + indexX;
  int transposedIndex = tindexY * N + tindexX;

  sharedMemory[localIndexX][localIndexY] = input[index];

  __syncthreads();

  output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
}

template<typename T>
T div_round_up(T a, T b) {
  return (a + b - 1) / b;
}

int main(void) {
  int *a, *b;

  size_t size = N * N *sizeof(int);
  double start = 0.0, end = 0.0, elapsed = 0.0;

  cudaMallocManaged((void **)&a, size);
  cudaMallocManaged((void **)&b, size);

  dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE,1);
  dim3 gridSize(N/BLOCK_SIZE,N/BLOCK_SIZE,1);

  matrix_initialization<<<div_round_up(N,BLOCK_SIZE),BLOCK_SIZE>>>(a,size);

  start = omp_get_wtime();
    matrix_transpose_naive<<<gridSize,blockSize>>>(a,b);
    getLastCudaError("matrix_transpose_naive() kernel failed");
  end = omp_get_wtime();
  elapsed = end-start;
  // printf("GB/s : %lf\n", (size)/(elapsed*(1<<30)));

  matrix_transpose_shared<<<gridSize,blockSize>>>(a,b);
  getLastCudaError("matrix_transpose_shared() kernel failed");

  // printf("b[0]: %d\n", a[0]);

  matrix_transpose_shared_non_conflict<<<gridSize,blockSize>>>(a,b);
  getLastCudaError("matrix_transpose_shared_non_conflict() kernel failed");

  cudaFree(a);
  cudaFree(b);

  return 0;
}
