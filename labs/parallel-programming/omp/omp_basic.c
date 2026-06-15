#include <omp.h>
#include <stdio.h>
#define NUM_THREADS 4
#define NUM_END 8

int main() {
    omp_set_num_threads(7);
    printf("=============Thread num example=============\n");

    #pragma omp parallel
        {
        printf("Thread #: %d, Total Threads: %d, numprocs: %d\n",
        omp_get_thread_num(), omp_get_num_threads(), omp_get_num_procs());
        }
    
    printf("=============Parallel for example=============\n");

    int i;
    omp_set_num_threads(NUM_THREADS);
    
    #pragma omp parallel for
    for (i=0; i<NUM_END; i++)
        printf("Thread #: %d, Total Threads: %d, numprocs: %d\n",
        omp_get_thread_num(), omp_get_num_threads(), omp_get_num_procs());
}