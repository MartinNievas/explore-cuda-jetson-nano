#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<string.h>
#include<omp.h>

#define NUM_THREADS 256l

#define N (1l<<20)

struct Coefficients_AOS {
  int r;
  int b;
  int g;
  int hue;
  int saturation;
  int maxVal;
  int minVal;
  int finalVal;
};


__global__
void color_conversion(Coefficients_AOS*  data)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;


  int grayscale = (data[i].r + data[i].g + data[i].b)/data[i].maxVal;
  int hue_sat = data[i].hue * data[i].saturation / data[i].minVal;
  data[i].finalVal = grayscale*hue_sat; 
}

template<typename T>
T div_round_up(T a, T b) {
  return (a + b - 1) / b;
}

int main(int argc, char*argv[])
{

  float start = 0.0, end = 0.0, elapsed = 0.0;
  Coefficients_AOS* d_x;

  start = omp_get_wtime();
  cudaMalloc(&d_x, N*sizeof(Coefficients_AOS)); 

  color_conversion<<<div_round_up(N,NUM_THREADS),NUM_THREADS>>>(d_x);

  cudaFree(d_x);
  end = omp_get_wtime();
  elapsed = end-start;
  printf("Elapsed[ms]: %f\n",elapsed*1000.);

  return 0;
}
