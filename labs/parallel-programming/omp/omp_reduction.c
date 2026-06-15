#include <omp.h>
#include <stdio.h>

#define NUM_THREADS 4
#define END_NUM 1000000

int main() {
    long long i;
    long long sum = 0;
    int mult_res = 1;
    double start_time, end_time;
    omp_set_num_threads(NUM_THREADS);
    start_time = omp_get_wtime();

    // reduction sum
    printf("reduction sum\n");
    #pragma omp parallel for reduction(+:sum)
        for (i = 1; i<END_NUM; i++)
            sum += i;
    
    end_time = omp_get_wtime();
    printf("result: %lld\n", sum);
    printf("computation time: %lf\n", end_time-start_time);

    start_time = omp_get_wtime();
    // reduction mult
    printf("reduction mult\n");
    #pragma omp parallel for reduction(*:mult_res)
        for (i=0; i<12; i++)
        mult_res *= 2;
    
    end_time = omp_get_wtime();
    printf("result: %d\n", mult_res);
    printf("time elapsed: %lf\n", end_time-start_time);

    return 0;
}