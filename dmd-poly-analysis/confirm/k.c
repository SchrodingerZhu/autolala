#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static double *A,*B,*C,*Cd;
static void fill(double*p,long n){unsigned long long s=88172645463325252ULL;for(long i=0;i<n;i++){s^=s<<13;s^=s>>7;s^=s<<17;p[i]=((s>>11)&((1ULL<<53)-1))/(double)(1ULL<<53);}}
int main(int argc,char**argv){
 int N=atoi(argv[1]); const char*k=argv[2];
 A=malloc(8L*N*N);B=malloc(8L*N*N);C=calloc(N*N,8);fill(A,N*N);fill(B,N*N);
 if(!strcmp(k,"mm_naive"))
   for(int i=0;i<N;i++)for(int j=0;j<N;j++){double c=0;for(int kk=0;kk<N;kk++)c+=A[i*N+kk]*B[kk*N+j];C[i*N+j]=c;}
 else if(!strcmp(k,"mm_tiled")){int T=32;
   for(int ii=0;ii<N;ii+=T)for(int kk=0;kk<N;kk+=T)for(int jj=0;jj<N;jj+=T)
    for(int i=ii;i<ii+T&&i<N;i++)for(int k=kk;k<kk+T&&k<N;k++){double a=A[i*N+k];
     for(int j=jj;j<jj+T&&j<N;j++)C[i*N+j]+=a*B[k*N+j];}}
 else if(!strcmp(k,"syrk_naive"))   // C=A*A^T lower tri, accumulator stays resident (RD const)
   for(int i=0;i<N;i++)for(int j=0;j<=i;j++){double c=0;for(int kk=0;kk<N;kk++)c+=A[i*N+kk]*A[j*N+kk];C[i*N+j]=c;}
 else if(!strcmp(k,"mvt_naive")){    // x2 += A^T y2 : transposed sweep, RD~N
   double*x=calloc(N,8),*y=malloc(8L*N);fill(y,N);
   for(int i=0;i<N;i++){double s=0;for(int j=0;j<N;j++)s+=A[j*N+i]*y[j];x[i]=s;}
   volatile double z=x[0];(void)z;}
 else if(!strcmp(k,"mvt_interch")){  // interchanged: contiguous A
   double*x=calloc(N,8),*y=malloc(8L*N);fill(y,N);
   for(int j=0;j<N;j++){double yj=y[j];for(int i=0;i<N;i++)x[i]+=A[j*N+i]*yj;}
   volatile double z=x[0];(void)z;}
 volatile double s=C[0];(void)s; return 0;}
