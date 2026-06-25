#include <stdio.h>
#include <stdlib.h>
// rank-2 memref descriptor (MLIR C-interface ABI)
typedef struct { double *alloc,*align; long off, size[2], stride[2]; } MR2;
extern void _mlir_ciface_kernel(MR2*,MR2*,MR2*,long);
static MR2 mk(double*p,long r,long c){ MR2 m={p,p,0,{r,c},{c,1}}; return m; }
int main(int argc,char**argv){
  long N=argc>1?atol(argv[1]):512;
  double*A=malloc(8L*N*N),*Bm=malloc(8L*N*N),*C=malloc(8L*N*N);
  unsigned long long s=88172645463325252ULL;
  for(long i=0;i<N*N;i++){s^=s<<13;s^=s>>7;s^=s<<17;A[i]=((s>>11)&((1ULL<<53)-1))/(double)(1ULL<<53);
                          s^=s<<13;s^=s>>7;s^=s<<17;Bm[i]=((s>>11)&((1ULL<<53)-1))/(double)(1ULL<<53); C[i]=0;}
  MR2 a=mk(A,N,N),b=mk(Bm,N,N),c=mk(C,N,N);
  _mlir_ciface_kernel(&a,&b,&c,N);
  volatile double sink=C[0];(void)sink;
  if(argc>2){FILE*f=fopen(argv[2],"wb");fwrite(C,8,N*N,f);fclose(f);}
  return 0;
}
