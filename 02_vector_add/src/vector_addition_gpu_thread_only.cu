#include<stdio.h>
#include<stdlib.h>
#include<assert.h>

#define N 1024

void host_add(int * const a, int * const b, int * const c) {
  for(int idx=0;idx<N;idx++)
    c[idx] = a[idx] + b[idx];
}

__global__
void device_add(int * const __restrict__ a,
int * const __restrict__ b,
int * const __restrict__ c) {

  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void fill_array(int * const data) {
  for(int idx=0;idx<N;idx++)
    data[idx] = idx;
}

void print_output(int *a, int *b, int*c) {
  for(int idx=0;idx<N;idx++)
    assert(c[idx] == a[idx]+b[idx]);
}

int main(void) {
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;

  int size = N * sizeof(int);

  a = (int *)malloc(size); fill_array(a);
  b = (int *)malloc(size); fill_array(b);
  c = (int *)malloc(size);

  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  device_add<<<1,N>>>(d_a,d_b,d_c);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  print_output(a,b,c);

  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}
