#include <cuda.h>
#include <stdio.h>
#include <assert.h>

#include "helper_cuda.h"

#define N (1L<<14)
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
  float *d_a=NULL, *d_b=NULL, *d_c=NULL, *d_d=NULL;

  a = (float *)malloc( N*sizeof(float));
  b = (float *)malloc( N*sizeof(float));
  c = (float *)malloc( N*sizeof(float));
  d = (float *)malloc( N*sizeof(float));

  for(size_t i = 0; i < N ; i++)
    a[i] = b[i] = c[i] = d[i] = 0.0;

  checkCudaErrors(cudaMalloc((void **)&d_a, N*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_b, N*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_c, N*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_d, N*sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_c, c, N*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_d, d, N*sizeof(float), cudaMemcpyHostToDevice));

  set<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_a,d_b,d_c,d_d);
  getLastCudaError("set() kernel failed");
  ma4<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_a,d_b,d_c,d_d);
  getLastCudaError("ma4() kernel failed");
  checkCudaErrors(cudaDeviceSynchronize());
  cudaMemcpy(d, d_d, N*sizeof(float), cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaDeviceSynchronize());

  for (size_t i=0; i<N; ++i)
    if (d[i] != a[i]*b[i]+c[i]) {
      printf("%d, %f!=%f*%f+%f\n", i, d[i], a[i], b[i], c[i]);
      break;
    }

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
  return 0;
}
