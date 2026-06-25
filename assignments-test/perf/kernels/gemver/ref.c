void kernel(int N, double* A, double* u1, double* v1, double* u2, double* v2, double* w, double* x, double* y, double* z){
 for(int i=0;i<N;i++) for(int j=0;j<N;j++) A[i*N+j]+=u1[i]*v1[j]+u2[i]*v2[j];
 for(int i=0;i<N;i++){ double xi=x[i]; for(int j=0;j<N;j++) xi+=1.1*A[j*N+i]*y[j]; x[i]=xi; }
 for(int i=0;i<N;i++) x[i]+=z[i];
 for(int i=0;i<N;i++){ double wi=0; for(int j=0;j<N;j++) wi+=1.2*A[i*N+j]*x[j]; w[i]=wi; }
}
