#include <stdio.h>
#include <stdlib.h>
// Usage:  ./bin N            -> run kernel only (for hyperfine timing; no I/O)
//         ./bin N out.bin    -> also dump output array(s) as raw float64 (for allclose)
void kernel(int N, double* A, double* u1, double* v1, double* u2, double* v2, double* w, double* x, double* y, double* z);
int main(int argc,char**argv){
  int N = argc>1?atoi(argv[1]):4096;
  double* A=malloc(sizeof(double)*(long)N*N);
  double* u1=malloc(sizeof(double)*(long)N);
  double* v1=malloc(sizeof(double)*(long)N);
  double* u2=malloc(sizeof(double)*(long)N);
  double* v2=malloc(sizeof(double)*(long)N);
  double* w=malloc(sizeof(double)*(long)N);
  double* x=malloc(sizeof(double)*(long)N);
  double* y=malloc(sizeof(double)*(long)N);
  double* z=malloc(sizeof(double)*(long)N);
  unsigned long long __rng=88172645463325252ULL;
  #define FILL(ptr,cnt) for(long i=0;i<(cnt);i++){__rng^=__rng<<13;__rng^=__rng>>7;__rng^=__rng<<17;(ptr)[i]=((__rng>>11)&((1ULL<<53)-1))/(double)(1ULL<<53);}
  #define ZERO(ptr,cnt) for(long i=0;i<(cnt);i++)(ptr)[i]=0.0;
  FILL(A,(long)N*N);
  FILL(u1,(long)N);
  FILL(v1,(long)N);
  FILL(u2,(long)N);
  FILL(v2,(long)N);
  ZERO(w,(long)N);
  ZERO(x,(long)N);
  FILL(y,(long)N);
  FILL(z,(long)N);
  kernel(N, A, u1, v1, u2, v2, w, x, y, z);
  // force the result to be materialized so timing runs can't be optimized away
  volatile double sink = w[0]; (void)sink;
  if(argc>2){ FILE*f=fopen(argv[2],"wb");
    fwrite(w,sizeof(double),(long)N,f);
    fclose(f); }
  return 0;
}
