void kernel(int N, double* A, double* x, double* y, double* tmp){
 for(int i=0;i<N;i++){ double t=0; for(int j=0;j<N;j++) t+=A[i*N+j]*x[j]; tmp[i]=t; }
 for(int i=0;i<N;i++) for(int j=0;j<N;j++) y[j]+=A[i*N+j]*tmp[i];
}
