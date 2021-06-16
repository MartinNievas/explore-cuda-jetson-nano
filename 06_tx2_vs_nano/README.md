# vector addition

Vector addition kernels
- block and thread

## How to compile
```bash
$ sh compile.sh
```

Executables are generated under: `build/bin/`


## Compare

|size|jetson nano | jetson tx2|
|----|------------|-----------|
| | | |

# Jetson Nano architecture

Output of [deviceQuery](https://github.com/NVIDIA/cuda-samples):

```
Device 0: "NVIDIA Tegra X1"
  CUDA Driver Version / Runtime Version          10.2 / 10.2
  CUDA Capability Major/Minor version number:    5.3
  Total amount of global memory:                 3964 MBytes (4156780544 bytes)
  ( 1) Multiprocessors, (128) CUDA Cores/MP:     128 CUDA Cores
  GPU Max Clock rate:                            922 MHz (0.92 GHz)
  Memory Clock rate:                             13 Mhz
  Memory Bus Width:                              64-bit
  L2 Cache Size:                                 262144 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 32768
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            Yes
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            No
  Supports Cooperative Kernel Launch:            No
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
```

### BLOCK_SIZE 256

```bash
$ sudo /usr/local/cuda-10.2/bin/nvprof --unified-memory-profiling off  bin/ma4
==3213== NVPROF is profiling process 3213, command: bin/ma4
16843775, 16843776.000000!=65795.000000*255.000000+66050.000000
==3213== Profiling application: bin/ma4
```

#### `define N (1L<<25)`
|            Type | Time(%)|      Time|     Calls|       Avg|       Min|       Max|  Name|
|---|---|---|---|---|---|---|---|
| GPU activities:   |51.06%  |28.724ms         |1  |28.724ms  |28.724ms  |28.724ms  |ma4(float*, float*, float*, float*)|
|                   |48.94%  |27.531ms         |1  |27.531ms  |27.531ms  |27.531ms  |set(float*, float*, float*, float*)|
|      API calls:   |87.21%  |656.04ms         |4  |164.01ms  |82.187ms  |394.92ms  |cudaMallocManaged|
|                   | 7.51%  |56.509ms         |1  |56.509ms  |56.509ms  |56.509ms  |cudaDeviceSynchronize|
|                   | 5.23%  |39.359ms         |4  |9.8398ms  |9.1791ms  |11.067ms  |cudaFree|
|                   | 0.03%  |209.47us         |2  |104.73us  |59.542us  |149.93us  |cudaLaunchKernel|

#### `define N (1L<<26)`
|            Type | Time(%)|      Time|     Calls|       Avg|       Min|       Max|  Name|
|---|---|---|---|---|---|---|---|
| GPU activities:   |50.85%  |57.188ms         |1  |57.188ms  |57.188ms  |57.188ms  |ma4(float*, float*, float*, float*)|
|                   |49.15%  |55.284ms         |1  |55.284ms  |55.284ms  |55.284ms  |set(float*, float*, float*, float*)|
|      API calls:   |80.81%  |880.17ms         |4  |220.04ms  |85.354ms  |489.17ms  |cudaMallocManaged|
|                   |10.35%  |112.70ms         |1  |112.70ms  |112.70ms  |112.70ms  |cudaDeviceSynchronize|
|                   | 8.81%  |95.910ms         |4  |23.978ms  |14.474ms  |32.765ms  |cudaFree|
|                   | 0.02%  |215.47us         |2  |107.73us  |61.838us  |153.63us  |cudaLaunchKernel|


# Jetson TX2 architecture

```bash
Device 0: "NVIDIA Tegra X2"
  CUDA Driver Version / Runtime Version          10.2 / 10.2
  CUDA Capability Major/Minor version number:    6.2
  Total amount of global memory:                 7858 MBytes (8240222208 bytes)
  ( 2) Multiprocessors, (128) CUDA Cores/MP:     256 CUDA Cores
  GPU Max Clock rate:                            1300 MHz (1.30 GHz)
  Memory Clock rate:                             1300 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 524288 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 32768
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            Yes
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.2, CUDA Runtime Version = 10.2, NumDevs = 1
Result = PASS
```

### BLOCK_SIZE 256

```bash
=27208== NVPROF is profiling process 27208, command: bin/ma4
==27208== Warning: Unified Memory Profiling is not supported on the underlying platform. System requirements for unified memory can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements
16843775, 16843776.000000!=65795.000000*255.000000+66050.000000
==27208== Profiling application: bin/ma4
```

#### `define N (1L<<25)`
|            Type | Time(%)|      Time|     Calls|       Avg|       Min|       Max|  Name|
|---|---|---|---|---|---|---|---|
| GPU activities: |  53.29%|  18.540ms|         1|  18.540ms|  18.540ms|  18.540ms|  set(float*, float*, float*, float*)|
|                 |  46.71%|  16.250ms|         1|  16.250ms|  16.250ms|  16.250ms|  ma4(float*, float*, float*, float*)|
|      API calls: |  80.44%|  353.10ms|         4|  88.276ms|  7.3051ms|  330.82ms|  cudaMallocManaged|
|                 |  11.47%|  50.357ms|         4|  12.589ms|  12.350ms|  13.043ms|  cudaFree|
|                 |   8.01%|  35.142ms|         1|  35.142ms|  35.142ms|  35.142ms|  cudaDeviceSynchronize|

#### `define N (1L<<26)`
|            Type | Time(%)|      Time|     Calls|       Avg|       Min|       Max|  Name|
|---|---|---|---|---|---|---|---|
|GPU activities:   |52.84%  |35.420ms         |1  |35.420ms  |35.420ms  |35.420ms  |set(float*, float*, float*, float*)|
|                  | 47.16% | 31.611ms        | 1 | 31.611ms | 31.611ms | 31.611ms | ma4(float*, float*, float*, float*) |
|      API calls:  | 74.67% | 474.30ms        | 4 | 118.57ms | 14.186ms | 356.78ms | cudaMallocManaged |
|                  | 14.65% | 93.073ms        | 4 | 23.268ms | 20.162ms | 28.084ms | cudaFree|
|                  | 10.61% | 67.395ms        | 1 | 67.395ms | 67.395ms | 67.395ms | cudaDeviceSynchronize |




