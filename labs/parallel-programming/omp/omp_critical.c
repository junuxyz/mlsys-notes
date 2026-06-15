#include <omp.h>
#include <stdio.h>

#define NUM_THREADS 4
#define END_NUM 1000000

int main() {
    long long i, j;
    long long sum = 0;
    long long reduct_sum = 0;
    double start, end, reduct_start, reduct_end;

    omp_set_num_threads(NUM_THREADS);
    
    printf("=========omp critical=========\n");
    start = omp_get_wtime();

    #pragma omp parallel for 
        for (i=1; i<END_NUM; i++)
            #pragma omp critical
            sum += i;
    
    end = omp_get_wtime();

    printf("result: %lld\n", sum);
    printf("elapsed time: %lf\n", end - start);

    reduct_start = omp_get_wtime();

    #pragma omp parallel for reduction(+:reduct_sum)
    for (j=1; j<END_NUM; j++)
        reduct_sum += j;

    reduct_end = omp_get_wtime();

    printf("=========omp reduction=========\n");
    printf("result: %lld\n", reduct_sum);
    printf("elapsed time: %lf\n", reduct_end - reduct_start);
    printf("===============================\n");

    printf("performance gain: x%lf\n",
    (end - start)/(reduct_end - reduct_start));

    return 0;
}