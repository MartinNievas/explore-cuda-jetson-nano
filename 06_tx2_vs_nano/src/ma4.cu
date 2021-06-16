#include <cuda.h>
#include <stdio.h>
#include <assert.h>

#include "helper_cuda.h"

#define N (1L<<26)
#define BLOCK_SIZE 256


__global__ void set(float *a, float *b, float *c, float *d) {
  unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
  a[gtid] = (float)blockIdx.x;
  b[gtid] = (float)threadIdx.x;
  c[gtid] = (float)threadIdx.x+blockIdx.x;
  d[gtid] = (float)threadIdx.x*blockIdx.x;;
}

__global__ void ma4(float *a, float *b, float *c, float *d) {
  unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
  d[gtid] = a[gtid]*b[gtid]+c[gtid];
}

int main(void)
{
  float *a=NULL, *b=NULL, *c=NULL, *d=NULL;
  checkCudaErrors(cudaMallocManaged(&a, N*sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&b, N*sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&c, N*sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&d, N*sizeof(float)));

  set<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(a,b,c,d);
  getLastCudaError("set() kernel failed");
  ma4<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(a,b,c,d);
  getLastCudaError("ma4() kernel failed");
  checkCudaErrors(cudaDeviceSynchronize());

  for (size_t i=0; i<N; ++i)
    if (d[i] != a[i]*b[i]+c[i]) {
      printf("%d, %f!=%f*%f+%f\n", i, d[i], a[i], b[i], c[i]);
      break;
    }

  cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(d);
  return 0;
}
