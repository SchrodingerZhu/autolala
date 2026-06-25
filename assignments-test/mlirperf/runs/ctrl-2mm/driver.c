#include <stdio.h>
#include <stdlib.h>
typedef struct{double*a,*b;long o,s[1],t[1];}MR1;
typedef struct{double*a,*b;long o,s[2],t[2];}MR2;
static MR1 mk1(double*p,long n){MR1 m={p,p,0,{n},{1}};return m;}
static MR2 mk2(double*p,long r,long c){MR2 m={p,p,0,{r,c},{c,1}};return m;}
extern void _mlir_ciface_kernel(void*,void*,void*,void*,void*,long);
int main(int argc,char**argv){
  long N=argc>1?atol(argv[1]):512;
  unsigned long long s=88172645463325252ULL;
  #define R s^=s<<13;s^=s>>7;s^=s<<17
  #define U ((s>>11)&((1ULL<<53)-1))/(double)(1ULL<<53)
  double*A=malloc(8L*N*N);
  double*B=malloc(8L*N*N);
  double*C=malloc(8L*N*N);
  double*D=malloc(8L*N*N);
  double*T=malloc(8L*N*N);
  for(long i=0;i<N*N;i++){R;A[i]=U;}
  for(long i=0;i<N*N;i++){R;B[i]=U;}
  for(long i=0;i<N*N;i++){R;C[i]=U;}
  for(long i=0;i<N*N;i++){R;D[i]=U;}
  for(long i=0;i<N*N;i++)T[i]=0.0;
  MR2 dA=mk2(A,N,N);
  MR2 dB=mk2(B,N,N);
  MR2 dC=mk2(C,N,N);
  MR2 dD=mk2(D,N,N);
  MR2 dT=mk2(T,N,N);
  _mlir_ciface_kernel(&dA,&dB,&dC,&dD,&dT,N);
  volatile double sink=D[0];(void)sink;
  if(argc>2){FILE*f=fopen(argv[2],"wb");fwrite(D,8,N*N,f);fclose(f);}
  return 0;
}
