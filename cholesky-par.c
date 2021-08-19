/**
Author: CS19S018
Code for: computing the Lower triangular matrix(L) by using the Cholesky decomposition for the problem 2 using OpenACC parallel implementation.
Status: 
complete: Yes,
compiles: Yes,
Runs: Yes,
Runs-and-gives-correct-result: Yes.

Compile:
pgcc -acc -Minfo=accel cholesky-par.c -lm
Run:
./a.out (num-of-gangs) > output.txt
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h> 

#define TYPE		double
#define N		10		// Matrix size NxN (can be changed)
#define SMALLVALUE	0.001
#define BIGVALUE	DBL_MAX
#define SMALLNEGVAL	DBL_MIN

// Matrix printing 
void printMat(TYPE a[][N]) {
	for (int ii = 0; ii < N; ++ii) {
		for (int jj = 0; jj < N; ++jj)
			printf("%.2f ", a[ii][jj]);
		printf("\n");
	}
	printf("\n\n");
}

// Performing the Cholesky-Crout decompostion, Note: Input matrix 'a' should symmetric positive definite.

int main(int argc, char **argv) {
	
	// Declaring matrix 'a'
	TYPE a[N][N];	

	// Declaring matrix 'L'
	TYPE L[N][N];

	// variable to compute the time
	struct timeval tstart, tend;

	int num_threads = 0;

	if(argc == 1){
  		printf("please enter the valid number of gangs\n\n");
		return 1;
  	}
	else{
 	  num_threads = atoi(argv[1]);
  	}

	//printMat(a);
	//gettimeofday(&t1, 0);
	gettimeofday(&tstart, 0);

	#pragma acc data create(a,L) copyout(L)
	{
	//cholesky(a,L,num_threads);
	TYPE t1;
	TYPE temp;

	//zero(a,num_threads);
	// Initialize all the values in 'a' matrix to zero
	#pragma acc parallel loop present(a[0:N][0:N]) collapse(2) num_gangs(num_threads)
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			a[i][j] = 0.0;
		}
	}	

	//zero(L,num_threads);
	// Initialize all the values in 'L' matrix to zero
	#pragma acc parallel loop present(L[0:N][0:N]) collapse(2) num_gangs(num_threads)
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			L[i][j] = 0.0;
		}
	}

	//init(a,num_threads);
	// set the values for matrix 'a' such that the resultant matrix is symmetic and positive definite
	#pragma acc parallel loop present(a[0:N][0:N]) collapse(2) num_gangs(num_threads)
	for (int ii = 0; ii < N; ++ii)
		for (int jj = 0; jj < N && jj < ii; ++jj)
			a[ii][jj] = a[jj][ii] = (ii * 1 + jj * 1) / (float)N / N;

	#pragma acc parallel loop present(a[0:N][0:N]) num_gangs(num_threads)
	for(int i=0;i<N;i++){
		//a[i][i] = 0.0;
		TYPE t = 0.0;
		#pragma acc loop reduction(+:t) 
		for(int j=0;j<N;j++){
				t += a[i][j];
		}
		a[i][i] = 0.1 + t;
	}
	
	// Perform the cholesky decomposition based on eqaution 6 given in PDF

	// Computing values column by column instead of row by row for parallelization purpose
	// This is because the value to be compute in each column depends only on the values of the previous column and the value present at the diagonal of the current column
	// Hence column by column computation is prefered..
	for (int jj = 0; jj < N; ++jj) {
	
		#pragma acc parallel loop present(L[0:N][0:N],a[0:N][0:N]) num_gangs(num_threads)
		for (int ii = 0; ii < 1; ++ii){

			temp=a[jj][jj];
			// Loop is parallelizable using reduction
			#pragma acc loop reduction(+:temp)
			for (int kk = 0; kk < jj; ++kk){
				temp += -L[jj][kk] * L[jj][kk];						
			}
			L[jj][jj] = (temp < SMALLNEGVAL ? SMALLNEGVAL : temp);				//check for underflow (multiplication value outside the range of double)
	
			if (L[jj][jj] >= 0)
				L[jj][jj] = sqrt(L[jj][jj]);
		}
	
		// Loop is parallelizable
		#pragma acc parallel loop present(L[0:N][0:N],a[0:N][0:N]) num_gangs(num_threads)
		for (int ii = jj+1; ii < N; ++ii) {
			t1=a[ii][jj];
			// Loop is parallelizable using reduction
			#pragma acc loop reduction(+:t1)
			for (int kk = 0; kk < jj; ++kk){
				t1 += (-L[ii][kk] * L[jj][kk]);					
			}
			L[ii][jj] = (t1 > BIGVALUE ? BIGVALUE : t1) ;				//check for overflow (multiplication value outside the range of double)
			L[ii][jj] = (t1 < SMALLNEGVAL ? SMALLNEGVAL : t1);				//check for underflow (multiplication value outside the range of double)

			L[ii][jj] /= (L[jj][jj] > SMALLVALUE ? L[jj][jj] : 1);			//checked for underflow (divide by zero)
		}

	}
	
	// cholesky decomposition end
	
	}

	//gettimeofday(&t2, 0);
	gettimeofday(&tend, 0);

	// Printing has to be in serial
	printMat(L);

	double time = (1000000.0*(tend.tv_sec-tstart.tv_sec) + tend.tv_usec-tstart.tv_usec)/1000.0; // Time taken
        printf("\nTime taken by cholesky decomposition to execute on GPU is: %.6f ms\n\n", time);

	return 0;
}

