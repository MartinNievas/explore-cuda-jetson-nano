#include<stdio.h>
#include<stdlib.h>
#include <assert.h>

#define N 1024

__global__
void device_add( int * const __restrict__ a,
int * const __restrict__ b,
int * const __restrict__ c) {
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__
void fill_array(int *const __restrict__ data) {
  data[blockIdx.x] = blockIdx.x;
}

__global__
void check_addition( int * const __restrict__ a,
int * const __restrict__ b,
int * const __restrict__ c) {
    assert(c[blockIdx.x] == (a[blockIdx.x]+b[blockIdx.x]));
}

int main(void) {
  int *a, *b, *c;

  int size = N * sizeof(int);

  // Alloc space for device a, b, c
  cudaMallocManaged((void **)&a, size);
  cudaMallocManaged((void **)&b, size);
  cudaMallocManaged((void **)&c, size);

  fill_array<<<N,1>>>(a);
  fill_array<<<N,1>>>(b);

  device_add<<<N,1>>>(a,b,c);

  check_addition<<<N,1>>>(a,b,c);

  cudaFree(a); cudaFree(b); cudaFree(c);



  return 0;
}
