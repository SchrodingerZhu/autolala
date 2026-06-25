#include <stdio.h>
#include <stdlib.h>
// Usage:  ./bin N            -> run kernel only (for hyperfine timing; no I/O)
//         ./bin N out.bin    -> also dump output array(s) as raw float64 (for allclose)
void kernel(int N, double* A, double* C);
int main(int argc,char**argv){
  int N = argc>1?atoi(argv[1]):1024;
  double* A=malloc(sizeof(double)*(long)N*N);
  double* C=malloc(sizeof(double)*(long)N*N);
  unsigned long long __rng=88172645463325252ULL;
  #define FILL(ptr,cnt) for(long i=0;i<(cnt);i++){__rng^=__rng<<13;__rng^=__rng>>7;__rng^=__rng<<17;(ptr)[i]=((__rng>>11)&((1ULL<<53)-1))/(double)(1ULL<<53);}
  #define ZERO(ptr,cnt) for(long i=0;i<(cnt);i++)(ptr)[i]=0.0;
  FILL(A,(long)N*N);
  FILL(C,(long)N*N);
  kernel(N, A, C);
  // force the result to be materialized so timing runs can't be optimized away
  volatile double sink = C[0]; (void)sink;
  if(argc>2){ FILE*f=fopen(argv[2],"wb");
    fwrite(C,sizeof(double),(long)N*N,f);
    fclose(f); }
  return 0;
}
