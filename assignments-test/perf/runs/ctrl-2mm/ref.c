void kernel(int N, double* A, double* B, double* C, double* D, double* tmp){
 for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double t=0;
  for(int k=0;k<N;k++) t+=1.1*A[i*N+k]*B[k*N+j]; tmp[i*N+j]=t; }
 for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double d=0.9*D[i*N+j];
  for(int k=0;k<N;k++) d+=tmp[i*N+k]*C[k*N+j]; D[i*N+j]=d; }
}
