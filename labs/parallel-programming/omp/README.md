# OpenMP

## Overview

OpenMP is a framework for shared memory parallel computing.

We use compiler directives that starts with `#pragma` where parallelism should be used.

Compiler directive means instruction to the compiler instead of normal program logic.

OpenMP is based on fork/join model, where the program starts as a single (master) thread. Note that master thread always has thread ID 0.
![[Pasted image 20260611212808.png]]
Figure. credit: an introduction into openmp

At designated parallel regions a pool of threads (or _workers_) is formed. In OpenMP, master and workers are also called as _team_. The threads execute in parallel across the parallel region. After all threads finishes, the master thread continues until the next parallel region.

Since OpenMP runtime creates and manages separate threads, it is transparent to the programmer - don't have to manually create and manage threads. So it's much easier to use OpenMP than lower level parallel libraries in general.

### OpenMP Components

There are three components of OpenMP which are compiler directives, environment variables, and runtime environment.

## Common Syntax

`omp_get_thread_num`: current thread index
`omp_get_num_threads`: size of the active team
`omp_get_max_threads`: maximum number of threads in given hardware
`omp_get_num_procs`: number of processors available
`omp_get_wtime`: elapsed wall clock time
`omp_get_wtick`: elapsed tick

`#pragma omp for`: executes multiple workers(threads) dividing the for loop

`#pragma omp for nowait`: without using `nowait`, each omp for section has implied barrier. If we use `nowait`, we remove the barrier so thread just moves on.


### avoiding race condition

When multiple threads try to access and update a variable (e.g. sum), we can use a critical section using `#pragma omp critical`. This allows only one thread to acquire a lock and access to the shared variable. However this may lead to synchronization overhead.

An alternative way is to use a reduction variable. If we use reduction, omp internally makes each thread accumulate their local operation and after all threads are finished, it does the final operation among local threads(e.g. sum or multiply etc. based on the context).

Performance comparison in [`omp_critical.cpp`](/omp/omp_critical.cpp):

result:
```
./omp_critical
=========omp critical=========
result: 499999500000
elapsed time: 0.022021
=========omp reduction=========
result: 499999500000
elapsed time: 0.000573
===============================
performance gain: x38.431065
```

### Shared and Private Variable

Shared variables are shared among all threads. By default, all variables declared outside a parallel block are shared except the loop index variable (e.g. in `for (int i; ...)`, i is private variable by default).

Private variables vary independently within threads. On entry and exit, values of private variables are undefined.

If `firstprivate` is used, variables are initialized to their value before the parallel region. If `lastprivate` is used, the value of variable after the loop is the value after the last iteration of the parallel section.

By default all variables declared inside the parallel region are private.

### Load balancing

For irregular workloads, load balancing is important for performance. The schedule clause supports various iteration scheduling algorithms.

There are four options:
![[Pasted image 20260611210020.png]]
Figure: credit (An introduction into OpenMP)


1. **static**
Static distributes iteration in blocks of size chunk over the threads in a round-robin fashion.

2. **dynamic**
execute fixed portions of work. The size of the portion is controlled by the chunk. When a thread finishes, it starts on the next portion of the work.

3. **guided (chunk)**
same dynamic behavior as dynamic but size of the work decreases incrementally.

4. **runtime**
iteration scheduling scheme is set at runtime through environment variable `OMP_SCHEDULE`

### SECTIONS

OpenMP `sections` assigns each independent code block (`section`) to one thread.

For example,

```cpp
NUM_THREADS 4
...
#pragma omp parallel default(none)\
		shared(n,a,b,c,d) private(i)
	{
		#pragma omp sections nowait
		{
			#pragma omp section // one thread; 3 threads remaining
			for (i=0; i<n-1; i++)
				b[i] = (a[i] + a[i+1])/2;
			
			#pragma omp section // another thread; 2 threads remaining
			for (i=0; i<n; i++)
				d[i] = 1.0/c[i];
		}
	}
```

in this code we delegate one thread to the first section and another thread to the next section.

