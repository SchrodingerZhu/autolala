void kernel(int N, double* A, double* x1, double* x2, double* y1, double* y2){
 for(int i=0;i<N;i++){ double s=x1[i]; for(int j=0;j<N;j++) s+=A[i*N+j]*y1[j]; x1[i]=s; }
 for(int i=0;i<N;i++){ double s=x2[i]; for(int j=0;j<N;j++) s+=A[j*N+i]*y2[j]; x2[i]=s; }
}
