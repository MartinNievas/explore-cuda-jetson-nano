#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<string.h>
#include<omp.h>

#define NUM_THREADS 256l

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
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int grayscale = (data.r[i] + data.g[i] + data.b[i])/data.maxVal[i];
  int hue_sat = data.hue[i] * data.saturation[i] / data.minVal[i];

  data.finalVal[i] = grayscale*hue_sat;
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

  int num_blocks = N/NUM_THREADS;

  color_conversion<<<div_round_up(N,NUM_THREADS),NUM_THREADS>>>(d_x);

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


