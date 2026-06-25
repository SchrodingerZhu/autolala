#include <stdio.h>
#include <stdlib.h>
// Usage:  ./bin N            -> run kernel only (for hyperfine timing; no I/O)
//         ./bin N out.bin    -> also dump output array(s) as raw float64 (for allclose)
void kernel(int N, double* A, double* p, double* r, double* q, double* s);
int main(int argc,char**argv){
  int N = argc>1?atoi(argv[1]):4096;
  double* A=malloc(sizeof(double)*(long)N*N);
  double* p=malloc(sizeof(double)*(long)N);
  double* r=malloc(sizeof(double)*(long)N);
  double* q=malloc(sizeof(double)*(long)N);
  double* s=malloc(sizeof(double)*(long)N);
  unsigned long long __rng=88172645463325252ULL;
  #define FILL(ptr,cnt) for(long i=0;i<(cnt);i++){__rng^=__rng<<13;__rng^=__rng>>7;__rng^=__rng<<17;(ptr)[i]=((__rng>>11)&((1ULL<<53)-1))/(double)(1ULL<<53);}
  #define ZERO(ptr,cnt) for(long i=0;i<(cnt);i++)(ptr)[i]=0.0;
  FILL(A,(long)N*N);
  FILL(p,(long)N);
  FILL(r,(long)N);
  ZERO(q,(long)N);
  ZERO(s,(long)N);
  kernel(N, A, p, r, q, s);
  // force the result to be materialized so timing runs can't be optimized away
  volatile double sink = q[0]; (void)sink;
  if(argc>2){ FILE*f=fopen(argv[2],"wb");
    fwrite(q,sizeof(double),(long)N,f);
    fwrite(s,sizeof(double),(long)N,f);
    fclose(f); }
  return 0;
}
