# Matrix transpose
Based on [this amazing blog's entry](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
The transpose of a matrix is an operator which flips a matrix over its diagonal; that is, it switches the row and column indices of the matrix A by producing another matrix.

The main idea is generate a matrix subdivision, and transpose this small block in shared memory, and store it on global memory.
![image from Nvida blog](https://developer.nvidia.com/blog/wp-content/uploads/2012/11/sharedTranspose-1024x409.jpg)


## How to compile
```bash
$ sh compile.sh
```

Executables are generated under: `build/bin/`
