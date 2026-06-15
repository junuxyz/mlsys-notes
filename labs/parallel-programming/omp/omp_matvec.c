#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 4
#define M 8
#define N 8

void mxv_row(int m, int n, double *a, double *b, double *c);

int main() {
    int i, j;
    double *Mat;
    double *Vec;
    double *Result_Vec;

    // init memory
    Mat = (double*)malloc(M*N*sizeof(double));
    Vec = (double*)malloc(N*sizeof(double));
    Result_Vec = (double*)malloc(N*sizeof(double));

    // init values
    for (i=0; i<M; i++) for (j=0; j<N; j++) Mat[i*N + j] = 1.0;
    for (i=0; i<N; i++) Vec[i] = 2.0;

    // mat vec mult
    mxv_row(M,N,Result_Vec,Mat,Vec);

    for (i=0; i<N; i++) printf("%1lf ", Result_Vec[i]);
    printf("\n");

    return 0;
}

void mxv_row(int m, int n, double *a, double *b, double *c) {
    int i, j;
    int sum;

    #pragma omp parallel for default(none) private(i,j,sum) shared(a,b,c,m,n)
    for (i=0; i<m; i++) {
        sum = 0;
        for (j=0; j<n; j++)
            sum += b[i*n+j] * c[j];
        a[i] = sum;
    }
}