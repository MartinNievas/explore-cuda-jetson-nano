#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <helper_functions.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on GPU
//! C = alpha * A * B + beta * C
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param C          matrix C as provided to device
//! @param N          height of matrix A and matrix C
//! @param M          width of matrix B and matrix C
//! @param K          width of matrix A and height of matrix C
//! @param alpha      scala value for matrix multiplication
//! @param beta       scala value for matrix summation with C
////////////////////////////////////////////////////////////////////////////////
  __global__ void
sgemm_gpu_kernel(float * const __restrict__ A, float * const __restrict__ B, float * const __restrict__ C,
int const N, int const M, int const K, float alpha, float beta)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0.f;
  for (int i = 0; i < K; ++i) {
    sum += A[row * K + i] * B[i * K + col];
  }

  C[row * M + col] = alpha * sum + beta * C[row * M + col];
}

void sgemm_gpu(float *A, float *B, float *C, int N, int M, int K, float alpha, float beta)
{
  dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);
  sgemm_gpu_kernel << < dimGrid, dimBlock >> > (A, B, C, N, M, K, alpha, beta);
}

__global__
void init(float * const data, const size_t size)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    data[index] = index;
  }
}

void performance_estimation(void(*sgemm)(float *, float *, float *, int, int, int, float, float),
    float * const A, float * const B, float *C, int N, int M, int K, float alpha, float beta)
{
  int test_iterations = 50;

  // Create timer
  StopWatchInterface *timer = 0;

  // initial start an operation as a warm start
  sgemm(A, B, C, N, M, K, alpha, beta);

  // Record the start event
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  ////////
  // Operation body
  ////////
  for (int i = 0; i < test_iterations; i++) {
    sgemm(A, B, C, N, M, K, alpha, beta);
  }

  // Waits for GPU operation finish and recored the time
  sdkStopTimer(&timer);

  // Compute and print the performance
  float operation_time = sdkGetAverageTimerValue(&timer);
  float operation_time_1_epoch = operation_time / test_iterations;

  printf("Operation Time= %.4f msec\n", operation_time_1_epoch);

  // cleanup
  sdkDeleteTimer(&timer);
}

template<typename T>
T div_round_up(T a, T b) {
  return (a + b - 1) / b;
}


int main()
{
  float *A, *B, *C;
  int N, M, K;
  float alpha = 2.f;
  float beta = 1.f;
  N = M = K = 1024;
  int threads_per_block = 128;

  // allocation of gpu linear memory space
  cudaMallocManaged((void **)&A, N * K * sizeof(float));
  cudaMallocManaged((void **)&B, K * M * sizeof(float));
  cudaMallocManaged((void **)&C, N * M * sizeof(float));

  // initialize randomized values for memory space
  init<<<div_round_up(N*N,threads_per_block),threads_per_block>>>(A, N * K);
  init<<<div_round_up(N*N,threads_per_block),threads_per_block>>>(B, K * M);
  init<<<div_round_up(N*N,threads_per_block),threads_per_block>>>(C, N * M);

  //sgemm_gpu(A, B, C, N, M, K, alpha, beta);
  performance_estimation(sgemm_gpu, A, B, C, N, M, K, alpha, beta);

  // terminates allocated gpu memory space
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
