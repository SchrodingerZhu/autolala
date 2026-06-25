void kernel(int N, double* data, double* cov, double* mean){
 for(int j=0;j<N;j++){ double m=0; for(int i=0;i<N;i++) m+=data[i*N+j]; mean[j]=m/N; }
 for(int i=0;i<N;i++) for(int j=0;j<N;j++) data[i*N+j]-=mean[j];
 for(int i=0;i<N;i++) for(int j=i;j<N;j++){ double c=0;
   for(int k=0;k<N;k++) c+=data[k*N+i]*data[k*N+j]; cov[i*N+j]=c; cov[j*N+i]=c; }
}
