#include <cuda.h>
#include <stdio.h>
#include <omp.h>
#include "helper_cuda.h"

__global__ void hello(void)
{
    printf("Hello thread %d, Block: %d\n", threadIdx.x, blockIdx.x);
}

int main(void)
{

  hello<<<2,3>>>();
  getLastCudaError("hello() kernel failed");
  cudaDeviceSynchronize();

  return 0;
}
