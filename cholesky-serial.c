/**
Author: CS19S018
Code for: computing the Lower triangular matrix(L) by using the Cholesky decomposition for the problem 2 using serial approach.
Status: 
complete: Yes,
compiles: Yes,
Runs: Yes,
Runs-and-gives-correct-result: Yes.

Compile:
gcc cholesky-serial.c -lm
Run:
./a.out > output.txt
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>


#define TYPE		double
#define N		10		// Matrix size NxN (can be changed)
#define SMALLVALUE	0.001

int main() {
	// variable to compute the time
	struct timeval tstart, tend;

	//TYPE a[N][N];
	// Declaring matrix 'a'
	TYPE **a = (TYPE **)malloc(N * sizeof(TYPE *));
    	
	for (int i=0; i<N; i++)
        	 a[i] = (TYPE *)malloc(N * sizeof(TYPE));	
	
	//TYPE L[N][N];
	// Declaring matrix 'L'
	TYPE **L = (TYPE **)malloc(N * sizeof(TYPE *));
    	
	for (int i=0; i<N; i++)
        	 L[i] = (TYPE *)malloc(N * sizeof(TYPE));	

	gettimeofday(&tstart, 0);

	//zero(a);
	// Initialize all the values in 'a' matrix to zero
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			a[i][j] = 0.0;
		}
	}
	
	//zero(L);	
	// Initialize all the values in 'L' matrix to zero
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			L[i][j] = 0.0;
		}
	}
	
	//init(a);
	// set the values for matrix 'a' such that the resultant matrix is symmetic and positive definite
	for (int ii = 0; ii < N; ++ii)
		for (int jj = 0; jj < N && jj < ii; ++jj)
			a[ii][jj] = a[jj][ii] = (ii * 1 + jj * 1) / (float)N / N;

	for(int i=0;i<N;i++){
		//a[i][i] = 0.0;
		TYPE t = 0.0;
		for(int j=0;j<N;j++){
				t += a[i][j];
		}
		a[i][i] = 0.1 + t;
	}
	//printMat(a);

	//cholesky(a,L);
	TYPE t1;
	TYPE temp;

	// Performing the cholesky decompostion, Note: Input matrix 'a' should symmetric positive definite.	
	for (int ii = 0; ii < N; ++ii) {
		for (int jj = 0; jj < ii; ++jj) {
			t1=a[ii][jj];
			for (int kk = 0; kk < jj; ++kk){
				t1 += (-L[ii][kk] * L[jj][kk]);
			}
			L[ii][jj] = t1;
			L[ii][jj] /= (L[jj][jj] > SMALLVALUE ? L[jj][jj] : 1);
			//a[ii][jj] /= a[jj][jj];	// divide by zero.
		}

		temp=a[ii][ii];
		for (int kk = 0; kk < ii; ++kk){
			temp += -L[ii][kk] * L[ii][kk];
		}
		L[ii][ii] = temp;
		if (L[ii][ii] >= 0)
			L[ii][ii] = sqrt(L[ii][ii]);
	}

	// cholesky decomposition ends

	gettimeofday(&tend, 0);

	double time = (1000000.0*(tend.tv_sec-tstart.tv_sec) + tend.tv_usec-tstart.tv_usec)/1000.0; // Time taken
        printf("\nTime taken by cholesky decomposition to execute on GPU is: %.6f ms\n\n", time);
	
	// Printing has to be in serial
	//printMat(L);
	
	// Print the results
	for (int ii = 0; ii < N; ++ii) {
		for (int jj = 0; jj < N; ++jj)
			printf("%.2f ", L[ii][jj]);
		printf("\n");
	}
	printf("\n\n");

	return 0;
}

