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
  double*x1=malloc(8L*N);
  double*x2=malloc(8L*N);
  double*y1=malloc(8L*N);
  double*y2=malloc(8L*N);
  for(long i=0;i<N*N;i++){R;A[i]=U;}
  for(long i=0;i<N;i++){R;x1[i]=U;}
  for(long i=0;i<N;i++){R;x2[i]=U;}
  for(long i=0;i<N;i++){R;y1[i]=U;}
  for(long i=0;i<N;i++){R;y2[i]=U;}
  MR2 dA=mk2(A,N,N);
  MR1 dx1=mk1(x1,N);
  MR1 dx2=mk1(x2,N);
  MR1 dy1=mk1(y1,N);
  MR1 dy2=mk1(y2,N);
  _mlir_ciface_kernel(&dA,&dx1,&dx2,&dy1,&dy2,N);
  volatile double sink=x1[0];(void)sink;
  if(argc>2){FILE*f=fopen(argv[2],"wb");fwrite(x1,8,N,f);fwrite(x2,8,N,f);fclose(f);}
  return 0;
}
