#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<string.h>
#include<omp.h>

#define NUM_THREADS 128l

#define N (1l<<20)

struct Coefficients_SOA {
  int* r;
  int* b;
  int* g;
  int* hue;
  int* saturation;
  int* maxVal;
  int* minVal;
  int* finalVal;
};


__global__
void color_conversion(Coefficients_SOA  data)
{
  // int i = blockIdx.x*blockDim.x + threadIdx.x;
  // int hue_sat = data.hue[i] * data.saturation[i] / data.minVal[i];
  // data.finalVal[i] = grayscale*hue_sat;

  unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  extern __shared__ float local_r[];
  extern __shared__ float local_g[];
  extern __shared__ float local_b[];
  extern __shared__ float local_maxVal[];
  extern __shared__ float local_minVal[];
  extern __shared__ float local_hue[];
  extern __shared__ float local_sat[];
  extern __shared__ float local_final[];
  // extern __shared__ float local_gray[];

  local_r[tid] = data.r[gtid];
  local_g[tid] = data.g[gtid];
  local_b[tid] = data.b[gtid];
  local_maxVal[tid] = data.maxVal[gtid];
  local_minVal[tid] = data.minVal[gtid];
  local_sat[tid] = data.saturation[gtid];
  local_hue[tid] = data.hue[gtid];
  __syncthreads();

    local_final[tid] = (local_r[tid] + local_g[tid] + local_b[tid])/local_maxVal[tid];
    local_final[tid] = local_final[tid] * local_hue[tid]* local_sat[tid] /local_minVal[tid];
  __syncthreads();

  data.finalVal[gtid] = local_final[tid];
  __syncthreads();

}

template<typename T>
T div_round_up(T a, T b) {
  return (a + b - 1) / b;
}

int main(int argc, char*argv[])
{

  float start = 0.0, end = 0.0, elapsed = 0.0;
  Coefficients_SOA d_x;

  start = omp_get_wtime();
  cudaMalloc(&d_x.r, N*sizeof(int));
  cudaMalloc(&d_x.g, N*sizeof(int));
  cudaMalloc(&d_x.b, N*sizeof(int));
  cudaMalloc(&d_x.hue, N*sizeof(int));
  cudaMalloc(&d_x.saturation, N*sizeof(int));
  cudaMalloc(&d_x.maxVal, N*sizeof(int));
  cudaMalloc(&d_x.minVal, N*sizeof(int));
  cudaMalloc(&d_x.finalVal, N*sizeof(int));

  color_conversion<<<div_round_up(N,NUM_THREADS),NUM_THREADS,9*NUM_THREADS>>>(d_x);

  cudaFree(d_x.r);
  cudaFree(d_x.g);
  cudaFree(d_x.b);
  cudaFree(d_x.hue);
  cudaFree(d_x.saturation);
  cudaFree(d_x.maxVal);
  cudaFree(d_x.maxVal);
  cudaFree(d_x.minVal);
  cudaFree(d_x.finalVal);
  end = omp_get_wtime();
  elapsed = end-start;
  printf("Elapsed[ms]: %f\n",elapsed*1000.);

  return 0;
}


