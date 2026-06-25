#include <stdio.h>
#include <stdlib.h>
// Usage:  ./bin N            -> run kernel only (for hyperfine timing; no I/O)
//         ./bin N out.bin    -> also dump output array(s) as raw float64 (for allclose)
void kernel(int N, double* A, double* x1, double* x2, double* y1, double* y2);
int main(int argc,char**argv){
  int N = argc>1?atoi(argv[1]):4096;
  double* A=malloc(sizeof(double)*(long)N*N);
  double* x1=malloc(sizeof(double)*(long)N);
  double* x2=malloc(sizeof(double)*(long)N);
  double* y1=malloc(sizeof(double)*(long)N);
  double* y2=malloc(sizeof(double)*(long)N);
  unsigned long long __rng=88172645463325252ULL;
  #define FILL(ptr,cnt) for(long i=0;i<(cnt);i++){__rng^=__rng<<13;__rng^=__rng>>7;__rng^=__rng<<17;(ptr)[i]=((__rng>>11)&((1ULL<<53)-1))/(double)(1ULL<<53);}
  #define ZERO(ptr,cnt) for(long i=0;i<(cnt);i++)(ptr)[i]=0.0;
  FILL(A,(long)N*N);
  FILL(x1,(long)N);
  FILL(x2,(long)N);
  FILL(y1,(long)N);
  FILL(y2,(long)N);
  kernel(N, A, x1, x2, y1, y2);
  // force the result to be materialized so timing runs can't be optimized away
  volatile double sink = x1[0]; (void)sink;
  if(argc>2){ FILE*f=fopen(argv[2],"wb");
    fwrite(x1,sizeof(double),(long)N,f);
    fwrite(x2,sizeof(double),(long)N,f);
    fclose(f); }
  return 0;
}
