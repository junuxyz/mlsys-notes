#include <omp.h>
#include <stdio.h>

#define NUM_THREADS 4
#define END_NUM 20

int main ()
{ 
	int i;
	double start_time, end_time;
	omp_set_num_threads(NUM_THREADS);
    printf("static scheduling\n");
    start_time = omp_get_wtime( );

	#pragma omp parallel for schedule(static,2)
	for (i = 1; i <= END_NUM; i++) {
		printf("%3d -- (%d/%d)\n",i, omp_get_thread_num(),omp_get_num_threads());
	}

	end_time = omp_get_wtime( );
	printf("time elapsed: %lfs\n",end_time-start_time);
    
    printf("dynamic scheduling\n");
    start_time = omp_get_wtime( );
    #pragma omp parallel for schedule(dynamic, 2)
	for (i = 1; i <= END_NUM; i++) {
		printf("%3d -- (%d/%d)\n",i, omp_get_thread_num(),omp_get_num_threads());
	}

	end_time = omp_get_wtime( );

	printf("time elapsed: %lfs\n",end_time-start_time);
    
    printf("guided scheduling\n");
    start_time = omp_get_wtime( );
    #pragma omp parallel for schedule(guided, 2)
	for (i = 1; i <= END_NUM; i++) {
		printf("%3d -- (%d/%d)\n",i, omp_get_thread_num(),omp_get_num_threads());
	}

	end_time = omp_get_wtime( );
	printf("time elapsed: %lfs\n",end_time-start_time);
	return 0;
}