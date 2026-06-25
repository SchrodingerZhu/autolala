#!/usr/bin/env python3
"""Generate the performance-experiment kernels.

For each kernel we emit:
  perf/kernels/<k>/ref.c     -- reference `kernel(...)` (naive, correct)
  perf/kernels/<k>/driver.c  -- main(): parse N, deterministic init, call kernel, print checksum
  perf/kernels/<k>/spec.md   -- the contract handed to the agent

The driver links against either ref.c or an agent's opt.c (identical signature),
so correctness == matching checksum and performance == hyperfine wall-clock.
All dims are the single size N. Constants alpha=1.1, beta=0.9 where relevant.
"""
import os, textwrap

# size class -> element count expression in N
SZ = {"V": "(long)N", "M": "(long)N*N", "T": "(long)N*N*N"}

KERNELS = {
"matmul": dict(
  size=(1024, 2048), default=1024,
  arrays=[("A","M","rand"),("B","M","rand"),("C","M","zero")],
  checksum=["C"],
  ref="""for (int i=0;i<N;i++) for(int j=0;j<N;j++){ double c=C[i*N+j];
    for(int k=0;k<N;k++) c+=A[i*N+k]*B[k*N+j]; C[i*N+j]=c; }""",
  desc="C += A*B  (all N x N row-major). Classic GEMM-like triple loop."),

"gemm": dict(
  size=(1024, 2048), default=1024,
  arrays=[("A","M","rand"),("B","M","rand"),("C","M","rand")],
  checksum=["C"],
  ref="""for(int i=0;i<N;i++) for(int j=0;j<N;j++) C[i*N+j]*=0.9;
   for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double c=C[i*N+j];
    for(int k=0;k<N;k++) c+=1.1*A[i*N+k]*B[k*N+j]; C[i*N+j]=c; }""",
  desc="GEMM: C = 0.9*C + 1.1*A*B  (all N x N)."),

"2mm": dict(
  size=(768, 1536), default=1024,
  arrays=[("A","M","rand"),("B","M","rand"),("C","M","rand"),("D","M","rand"),("tmp","M","zero")],
  checksum=["D"],
  ref="""for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double t=0;
    for(int k=0;k<N;k++) t+=1.1*A[i*N+k]*B[k*N+j]; tmp[i*N+j]=t; }
   for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double d=0.9*D[i*N+j];
    for(int k=0;k<N;k++) d+=tmp[i*N+k]*C[k*N+j]; D[i*N+j]=d; }""",
  desc="2mm: tmp = 1.1*A*B ; D = 0.9*D + tmp*C  (all N x N)."),

"3mm": dict(
  size=(768, 1536), default=1024,
  arrays=[("A","M","rand"),("B","M","rand"),("C","M","rand"),("D","M","rand"),
          ("E","M","zero"),("F","M","zero"),("G","M","zero")],
  checksum=["G"],
  ref="""for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double e=0;
    for(int k=0;k<N;k++) e+=A[i*N+k]*B[k*N+j]; E[i*N+j]=e; }
   for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double f=0;
    for(int k=0;k<N;k++) f+=C[i*N+k]*D[k*N+j]; F[i*N+j]=f; }
   for(int i=0;i<N;i++) for(int j=0;j<N;j++){ double g=0;
    for(int k=0;k<N;k++) g+=E[i*N+k]*F[k*N+j]; G[i*N+j]=g; }""",
  desc="3mm: E=A*B ; F=C*D ; G=E*F  (all N x N)."),

"mvt": dict(
  size=(4096, 8192), default=4096,
  arrays=[("A","M","rand"),("x1","V","rand"),("x2","V","rand"),("y1","V","rand"),("y2","V","rand")],
  checksum=["x1","x2"],
  ref="""for(int i=0;i<N;i++){ double s=x1[i]; for(int j=0;j<N;j++) s+=A[i*N+j]*y1[j]; x1[i]=s; }
   for(int i=0;i<N;i++){ double s=x2[i]; for(int j=0;j<N;j++) s+=A[j*N+i]*y2[j]; x2[i]=s; }""",
  desc="mvt: x1 += A*y1 ; x2 += A^T*y2  (A is N x N; note the transposed access in the 2nd loop)."),

"atax": dict(
  size=(4096, 8192), default=4096,
  arrays=[("A","M","rand"),("x","V","rand"),("y","V","zero"),("tmp","V","zero")],
  checksum=["y"],
  ref="""for(int i=0;i<N;i++){ double t=0; for(int j=0;j<N;j++) t+=A[i*N+j]*x[j]; tmp[i]=t; }
   for(int i=0;i<N;i++) for(int j=0;j<N;j++) y[j]+=A[i*N+j]*tmp[i];""",
  desc="atax: tmp = A*x ; y = A^T*tmp  (A is N x N)."),

"bicg": dict(
  size=(4096, 8192), default=4096,
  arrays=[("A","M","rand"),("p","V","rand"),("r","V","rand"),("q","V","zero"),("s","V","zero")],
  checksum=["q","s"],
  ref="""for(int i=0;i<N;i++){ double qi=0; for(int j=0;j<N;j++){ s[j]+=r[i]*A[i*N+j];
      qi+=A[i*N+j]*p[j]; } q[i]=qi; }""",
  desc="bicg: q = A*p and s = A^T*r, fused over the same A sweep (A is N x N)."),

"gemver": dict(
  size=(4096, 8192), default=4096,
  arrays=[("A","M","rand"),("u1","V","rand"),("v1","V","rand"),("u2","V","rand"),("v2","V","rand"),
          ("w","V","zero"),("x","V","zero"),("y","V","rand"),("z","V","rand")],
  checksum=["w"],
  ref="""for(int i=0;i<N;i++) for(int j=0;j<N;j++) A[i*N+j]+=u1[i]*v1[j]+u2[i]*v2[j];
   for(int i=0;i<N;i++){ double xi=x[i]; for(int j=0;j<N;j++) xi+=1.1*A[j*N+i]*y[j]; x[i]=xi; }
   for(int i=0;i<N;i++) x[i]+=z[i];
   for(int i=0;i<N;i++){ double wi=0; for(int j=0;j<N;j++) wi+=1.2*A[i*N+j]*x[j]; w[i]=wi; }""",
  desc="gemver: A += u1*v1^T + u2*v2^T ; x = 1.1*A^T*y + z ; w = 1.2*A*x  (A is N x N)."),

"gesummv": dict(
  size=(4096, 8192), default=4096,
  arrays=[("A","M","rand"),("B","M","rand"),("x","V","rand"),("y","V","zero"),("tmp","V","zero")],
  checksum=["y"],
  ref="""for(int i=0;i<N;i++){ double t=0,yy=0; for(int j=0;j<N;j++){ t+=A[i*N+j]*x[j];
      yy+=B[i*N+j]*x[j]; } tmp[i]=t; y[i]=1.1*t+0.9*yy; }""",
  desc="gesummv: y = 1.1*(A*x) + 0.9*(B*x)  (A,B are N x N)."),

"syrk": dict(
  size=(1024, 2048), default=1024,
  arrays=[("A","M","rand"),("C","M","rand")],
  checksum=["C"],
  ref="""for(int i=0;i<N;i++) for(int j=0;j<=i;j++) C[i*N+j]*=0.9;
   for(int i=0;i<N;i++) for(int k=0;k<N;k++){ double a=1.1*A[i*N+k];
     for(int j=0;j<=i;j++) C[i*N+j]+=a*A[j*N+k]; }""",
  desc="syrk: C = 0.9*C + 1.1*A*A^T, lower triangle only (A, C are N x N)."),

"doitgen": dict(
  size=(192, 384), default=256,
  arrays=[("A","T","rand"),("C4","M","rand"),("sum","V","zero")],
  checksum=["A"],
  ref="""for(int r=0;r<N;r++) for(int q=0;q<N;q++){
     for(int p=0;p<N;p++){ double s=0; for(int t=0;t<N;t++) s+=A[(r*N+q)*N+t]*C4[t*N+p]; sum[p]=s; }
     for(int p=0;p<N;p++) A[(r*N+q)*N+p]=sum[p]; }""",
  desc="doitgen: for each (r,q): A[r][q][:] = sum_t A[r][q][t]*C4[t][:]  (A is N x N x N, C4 is N x N). Memory is O(N^3) so N is modest."),

"covariance": dict(
  size=(1024, 2048), default=1024,
  arrays=[("data","M","rand"),("cov","M","zero"),("mean","V","zero")],
  checksum=["cov"],
  ref="""for(int j=0;j<N;j++){ double m=0; for(int i=0;i<N;i++) m+=data[i*N+j]; mean[j]=m/N; }
   for(int i=0;i<N;i++) for(int j=0;j<N;j++) data[i*N+j]-=mean[j];
   for(int i=0;i<N;i++) for(int j=i;j<N;j++){ double c=0;
     for(int k=0;k<N;k++) c+=data[k*N+i]*data[k*N+j]; cov[i*N+j]=c; cov[j*N+i]=c; }""",
  desc="covariance: mean-center columns of data (N x N), then cov = data^T*data (symmetric)."),
}

DRIVER_TMPL = """#include <stdio.h>
#include <stdlib.h>
// Usage:  ./bin N            -> run kernel only (for hyperfine timing; no I/O)
//         ./bin N out.bin    -> also dump output array(s) as raw float64 (for allclose)
void kernel(int N{params});
int main(int argc,char**argv){{
  int N = argc>1?atoi(argv[1]):{default};
{allocs}
  unsigned long long __rng=88172645463325252ULL;
  #define FILL(ptr,cnt) for(long i=0;i<(cnt);i++){{__rng^=__rng<<13;__rng^=__rng>>7;__rng^=__rng<<17;(ptr)[i]=((__rng>>11)&((1ULL<<53)-1))/(double)(1ULL<<53);}}
  #define ZERO(ptr,cnt) for(long i=0;i<(cnt);i++)(ptr)[i]=0.0;
{inits}
  kernel(N{args});
  // force the result to be materialized so timing runs can't be optimized away
  volatile double sink = {sink}[0]; (void)sink;
  if(argc>2){{ FILE*f=fopen(argv[2],"wb");
{dumps}
    fclose(f); }}
  return 0;
}}
"""

def gen(name, spec):
    d = f"kernels/{name}"
    os.makedirs(d, exist_ok=True)
    arrs = spec["arrays"]
    params = "".join(f", double* {a}" for a,_,_ in arrs)
    # ref.c
    body = textwrap.dedent("  void kernel(int N%s){\n   %s\n  }\n" % (params, spec["ref"].strip()))
    open(f"{d}/ref.c","w").write(body)
    # driver.c
    allocs = "\n".join(f"  double* {a}=malloc(sizeof(double)*{SZ[s]});" for a,s,_ in arrs)
    inits  = "\n".join((f"  FILL({a},{SZ[s]});" if init=="rand" else f"  ZERO({a},{SZ[s]});")
                       for a,s,init in arrs)
    args   = "".join(f", {a}" for a,_,_ in arrs)
    szof   = {a:s for a,s,_ in arrs}
    dumps  = "\n".join(f"    fwrite({c},sizeof(double),{SZ[szof[c]]},f);" for c in spec["checksum"])
    drv = DRIVER_TMPL.format(params=params, default=spec["default"], allocs=allocs,
                             inits=inits, args=args, dumps=dumps, sink=spec["checksum"][0])
    open(f"{d}/driver.c","w").write(drv)
    return name

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for k,v in KERNELS.items():
        gen(k,v)
    print("generated", len(KERNELS), "kernels:", " ".join(KERNELS))
