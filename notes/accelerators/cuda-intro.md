GPU is based on SIMT(Single Instruction, Multiple Threads). This means it executes instruction across multiple threads. Group of threads is controlled by control unit called SM.
SM executes a _warp_ which is a group of 32 threads.

SIMT is a variation of SIMD but there is an important distinction. SIMD deterministically executes one instruction. SIMT on the other hand, allows divergent workflow among threads in a group(e.g. if-else branch). However this does add performance penalty.

<p align="center">
  <img src="/assets/notes/cuda-intro/cuda-intro-1.png" width="620" />
  <br />
  <sub>Figure 1. Warp divergence causes threads in the same warp to follow different execution paths.</sub>
</p>

We will see what warp is later in the note.

## basic syntax

### `__global__`

`__global__` is a function declaration which
- runs on the device(GPU)
- is called from host(CPU) code

Internally, when compiling a `.cu` program, nvcc (Nvidia compiler) compiles device functions and standard host compiler (e.g. gcc) compiles host functions.

Functions using `__global__` should have `void` return type because it changes the device memory directly.

Example:
```c
__global__ void add(int *a, int *b, int *c) {
 // ...
}
```

It is important to note that we are sending the pointer  of device memory as arguments. This means we need to predefine and allocate memory address and space for the kernel before launching it.

### `__device__`

executed on the device and only callable from the device. Usually used as an internal helper function for kernel.

### `__host__`

executed on host and only callable from host. This is just C/C++ code but using a more explicit expression. `__device__` and `__host__` can be used together, though.

### `<<<>>>`

Triple angle brackets refers to a call from host to device code. This process is also called a _kernel launch_.

### `cudaMalloc`, `cudaMemcpy`, and `cudaFree`

These three predefined functions are all used for memory management for device.

`cudaMalloc` allocates the memory for device so it can read the inputs and write it in output.

`cudaMemCpy` copies host memory to device memory. This is usually related to inputs(arguments). See example in the typical processing flow below.


> [!question] Why do we use `(void**) &d_a` in `cudaMalloc` and `d_a` in `cudaMemCpy`?
> 
> This is because we are trying to change the memory address the pointer is directing in `cudaMalloc` while in `cudaMemCpy` we are trying to change the value of the memory address pointer is pointing at.


### `cudaMallocHost`, `malloc`, `cudaMallocManaged`

It is important to understand how these differ.

`cudaMalloc()` allocates memory in device memory, usually GPU DRAM. The host cannot directly dereference this pointer and can indirectly copy using `cudaMemcpy()`.

`malloc()` allocates normal pageable host memory in CPU memory. _pageable_ means it can be swapped.

`cudaMallocHost()` allocates pinned/page-locked host memory. Since this memory cannot be swapped out by the OS, GPU DMA(direct memory access) transfers can be faster and can support efficient asynchronous copies.

`cudaMallocManaged()` allocates unified memory. CPU and GPU can access using the same pointer so programmer does not need separate host/device allocations and explicit copies in simple cases. 

However, data may migrate between CPU and GPU memory at runtime, and page faults or migration overhead can make performance slower or less predictable than explicit `cudaMalloc()` + `cudaMemcpy()`.


### grid, block, and thread

<p align="center">
  <img src="/assets/notes/cuda-intro/cuda-intro-2.png" width="540" />
  <br />
  <sub>Figure 2. A CUDA kernel launches a grid composed of thread blocks, each containing multiple threads.</sub>
</p>

When launching a kernel, you can think of launching  **grid of thread blocks**. One kernel cannot execute multiple number of grids. Only one is allowed.

For example, if we use `add<<<1024,1024>>>`, we're launching a grid that has 1024 blocks in that grid and 1024 threads per block. In total this will lead to ~1 million threads.


> [!NOTE]
>
> You might ask:
> "what's the difference between `<<<a, b>>>` vs `<<<b, a>>>` (a != b)?" 
> 
> Yes, we are launching the same amount of threads overall but usually the number of threads per block or `blockDim` should be considered more carefully. This is because one warp cannot divide a single block and executed. Also there are maximum number of threads that can be constructed within a block, due to SM size.
> 
> In short, you can say gridDim is much more flexible to change while block size is more carefully set and block shape can influence the size.


Block

Thread

Syntax for grid, block, and dim can feel kind of 헷갈림 so I'll make it clear.

Think of index as the i'th local location and dimension as the size of the thread or block.

<p align="center">
  <img src="/assets/notes/cuda-intro/cuda-intro-3.png" width="700" />
  <br />
  <sub>Figure 3. <code>threadIdx</code> identifies a thread within its block, while <code>blockIdx</code> identifies the block within the grid.</sub>
</p>

### `threadIdx`
refers to which index it is **locally** inside a block. For every block, each `threadIdx` starts from 0 to the size of that block - 1. `threadIdx` can be composed of three dimensions `threadIdx.x`, `threadIdx.y`, `threadIdx.z` but in deep learning we usually end up using 1 dim only.

### `blockIdx`
refers to which block it is locally inside the single grid. Likewise `blockIdx` can be composed of three dimensions `blockIdx.x`, `blockIdx.y`, `blockIdx.z`.

### `blockDim`
refers to the number of threads each blocks have. This is identical to all blocks. Note that for Dim, it usually uses one dimension above what it's trying to describe (이걸 말이 되게 설명하고 싶은데...)

### `gridDim`
refers to the number of blocks a single grid has.

### ex1. vector sum

A typical example using vector sum would be:

```cpp
const int THREADS_PER_BLOCK = 1024

// 3. execute parallel program in kernel
__global__ void add(*d_a, *d_b, *d_c, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n)
		// does not return value; save it in d_c instead
		d_c[idx] = d_a[idx] + d_b[idx];
}

int main(void) {
	int a[N], b[N], c[N];
	int* d_a, d_b, d_c;
	
	// allocate space for device copies of a, b, c
	cudaMalloc((void**)&d_a, sizeof(int) * N)
	cudaMalloc((void**)&d_b, sizeof(int) * N)
	cudaMalloc((void**)&d_c, sizeof(int) * N)
	
	// setup input variable in host
	for (int i=0; i<N; i++) {
		a[i] = i;
		b[i] = i;
	}
	
	// 1. copy inputs to device
	cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
	// we don't need to copy c because it's going to be updated as output
	
	// 2. launch kernel with N blocks of 1024 threads
	add<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
	
	// 4. copy result back to host
	cudaMemcpy(c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost);
	
	// cleanup
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return 0;
}
```

### `__shared__`

shared data among multiple threads within a block. Since shared memory is extremely fast on-chip memory, if we need to access data from global memory multiple times, we can store it in shared memory and then access it to reduce the memory bottleneck.


### `__syncthreads()`

When threads cooperate, execution order and finish time is non-deterministic. In order to 보장 stability, we can use `__syncthreads()` which means synchronizing and ensuring all data is available at that point. This is usually used after launching the kernel.

## a typical processing flow

<p align="center">
  <img src="/assets/notes/cuda-intro/cuda-intro-4.png" width="540" />
  <br />
  <sub>Figure 4. A typical CUDA program allocates device memory, transfers inputs, launches a kernel, and copies results back to the host.</sub>
</p>



