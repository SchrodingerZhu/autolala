void kernel(int N, double* A, double* B, double* C, double* D, double* E, double* F, double* G){
 for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double e=0;
  for(int k=0;k<N;k++) e+=A[i*N+k]*B[k*N+j]; E[i*N+j]=e; }
 for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double f=0;
  for(int k=0;k<N;k++) f+=C[i*N+k]*D[k*N+j]; F[i*N+j]=f; }
 for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double g=0;
  for(int k=0;k<N;k++) g+=E[i*N+k]*F[k*N+j]; G[i*N+j]=g; }
}
