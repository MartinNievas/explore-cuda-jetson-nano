#include<stdio.h>
#include<stdlib.h>
#include <assert.h>

#define N 1024

__global__
void device_add( int * const __restrict__ a,
int * const __restrict__ b,
int * const __restrict__ c) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

__global__
void fill_array(int *const __restrict__ data) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  data[index] = index;
}

__global__
void check_addition( int * const __restrict__ a,
int * const __restrict__ b,
int * const __restrict__ c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  assert(c[index] == (a[index]+b[index]));
}

template<typename T>
T div_round_up(T a, T b) {
  return (a + b - 1) / b;
}

int main(void) {
  int *a, *b, *c;
  int threads_per_block=4;

  int size = N * sizeof(int);

  // Alloc space for device a, b, c
  cudaMallocManaged((void **)&a, size);
  cudaMallocManaged((void **)&b, size);
  cudaMallocManaged((void **)&c, size);

  fill_array<<<div_round_up(N,threads_per_block),threads_per_block>>>(a);
  fill_array<<<div_round_up(N,threads_per_block),threads_per_block>>>(b);
  device_add<<<div_round_up(N,threads_per_block),threads_per_block>>>(a,b,c);

  check_addition<<<div_round_up(N,threads_per_block),threads_per_block>>>(a,b,c);

  cudaFree(a); cudaFree(b); cudaFree(c);

  return 0;
}
