void kernel(int N, double* A, double* B, double* C){
 for(int i=0;i<N;i++) for(int j=0;j<N;j++) C[i*N+j]*=0.9;
 for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double c=C[i*N+j];
  for(int k=0;k<N;k++) c+=1.1*A[i*N+k]*B[k*N+j]; C[i*N+j]=c; }
}
