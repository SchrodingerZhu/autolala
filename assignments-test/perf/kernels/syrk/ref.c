void kernel(int N, double* A, double* C){
 for(int i=0;i<N;i++) for(int j=0;j<=i;j++) C[i*N+j]*=0.9;
 for(int i=0;i<N;i++) for(int k=0;k<N;k++){ double a=1.1*A[i*N+k];
   for(int j=0;j<=i;j++) C[i*N+j]+=a*A[j*N+k]; }
}
