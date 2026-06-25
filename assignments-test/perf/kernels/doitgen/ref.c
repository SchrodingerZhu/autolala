void kernel(int N, double* A, double* C4, double* sum){
 for(int r=0;r<N;r++) for(int q=0;q<N;q++){
   for(int p=0;p<N;p++){ double s=0; for(int t=0;t<N;t++) s+=A[(r*N+q)*N+t]*C4[t*N+p]; sum[p]=s; }
   for(int p=0;p<N;p++) A[(r*N+q)*N+p]=sum[p]; }
}
