void kernel(int N, double* A, double* B, double* x, double* y, double* tmp){
 for(int i=0;i<N;i++){ double t=0,yy=0; for(int j=0;j<N;j++){ t+=A[i*N+j]*x[j];
    yy+=B[i*N+j]*x[j]; } tmp[i]=t; y[i]=1.1*t+0.9*yy; }
}
