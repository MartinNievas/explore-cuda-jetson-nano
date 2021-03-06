#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include "helper_cuda.h"

#define N 1024

__global__
void device_add( int * const __restrict__ a,
int * const __restrict__ b,
int * const __restrict__ c) {

  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void fill_array(int *data) {
  for(int idx=0;idx<N;idx++)
    data[idx] = idx;
}

void check_addition(int *a, int *b, int*c) {
  for(int idx=0;idx<N;idx++)
    assert(c[idx] == (a[idx]+b[idx]));
}

int main(void) {
  int *a, *b, *c;
  int *d_a, *d_b, *d_c; // device copies of a, b, c

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


  device_add<<<N,1>>>(d_a,d_b,d_c);
  getLastCudaError("device_addevice_add() kernel failed");

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  check_addition(a,b,c);

  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);



  return 0;
}
