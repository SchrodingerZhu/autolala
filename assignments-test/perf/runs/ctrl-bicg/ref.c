void kernel(int N, double* A, double* p, double* r, double* q, double* s){
 for(int i=0;i<N;i++){ double qi=0; for(int j=0;j<N;j++){ s[j]+=r[i]*A[i*N+j];
    qi+=A[i*N+j]*p[j]; } q[i]=qi; }
}
