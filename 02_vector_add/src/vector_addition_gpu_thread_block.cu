#include<stdio.h>
#include<stdlib.h>
#include <assert.h>
#include "helper_cuda.h"

#define N 1024

__global__
void device_add( int * const __restrict__ a,
int * const __restrict__ b,
int * const __restrict__ c) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

void fill_array(int * const data) {
  for(int idx=0;idx<N;idx++)
    data[idx] = idx;
}

void print_output(int * const a, int * const b, int * const c) {
  for(int idx=0;idx<N;idx++)
    assert(c[idx] == (a[idx]+b[idx]));
}

template<typename T>
T div_round_up(T a, T b) {
  return (a + b - 1) / b;
}

int main(void) {
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;
  int threads_per_block=0;

  int size = N * sizeof(int);

  // Alloc space for host copies of a, b, c and setup input values
  a = (int *)malloc(size); fill_array(a);
  b = (int *)malloc(size); fill_array(b);
  c = (int *)malloc(size);

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  threads_per_block = 4;
  device_add<<<div_round_up(N,threads_per_block),threads_per_block>>>(d_a,d_b,d_c);
  getLastCudaError("device_add() kernel failed");

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  print_output(a,b,c);

  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}
