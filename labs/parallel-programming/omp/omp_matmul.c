#include <stdio.h>
#include <omp.h>

#define NUM_THREADS 8

#define NX 1000
#define NM NX
#define NY NX

int a[NX * NM];
int b[NM * NY];
int m[NY * NX];

#define A(i, n) a[(i) + NX * (n)]
#define B(n, j) b[(n) + NY * (j)]
#define M(i, j) m[(i) + NM * (j)]

void printMatrix(int* mat, int X, int Y) {
    int i, j;
    for (j = 0; j<Y; j++)
        for (i=0;i<X;i++)
            printf("%4d ", mat[i+j*X]);
    printf("\n");
}

int main() {
    int i, j, n;
    double t1, t2;

    omp_set_num_threads(NUM_THREADS);
    t1 = omp_get_wtime();

    #pragma omp parallel for default(shared) private(n, i)
    for (n=0;n<NM;n++)
        for (i=0;i<NX;i++)
            A(i, n) =3;
    
    #pragma omp parallel for default(shared) private(n, j)
    for (n=0;n<NM;n++)
        for (j=0;j<NM;j++)
            B(i, n) = 2;
    
    #pragma omp parallel for default(shared) private(i, j)
    for (j = 0; j < NY; j++)
	    for (i = 0; i < NX; i++)
	        M(i, j) = 0;

    // Matmul
    #pragma omp parallel for default(shared) private(i,j,n)
    for (j=0; j<NY; j++)
        for (n=0; n<NM; n++)
            for (i=0; i<NX; i++)
                M(i,j) += A(i,n) * B(n,j);

    t2 = omp_get_wtime();

    printMatrix(m, NX, NY);
    printf("computation time: %lf, using %d threads\n", t2-t1, NUM_THREADS);
    return 0;
}
