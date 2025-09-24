#define DATA_TYPE float
#define A_SIZE 16
#define B_SIZE 16
#define C_SIZE 16
#define D_SIZE 16
#define E_SIZE 16
#define F_SIZE 16
#define G_SIZE 16
#define H_SIZE 16
#define I_SIZE 16
#define J_SIZE 16
#define K_SIZE 16
#define L_SIZE 16

// Max-Cut quantum circuit: a,b,c,da,eb,fc,ghde,ijgf,klhj,i,k,l->
// Memory access pattern: quantum circuit tensor network contraction
void kernel_maxcut_quantum_circuit_pattern(DATA_TYPE a[A_SIZE], 
                                          DATA_TYPE b[B_SIZE], 
                                          DATA_TYPE c[C_SIZE], 
                                          DATA_TYPE da[D_SIZE][A_SIZE], 
                                          DATA_TYPE eb[E_SIZE][B_SIZE], 
                                          DATA_TYPE fc[F_SIZE][C_SIZE], 
                                          DATA_TYPE ghde[G_SIZE][H_SIZE][D_SIZE][E_SIZE], 
                                          DATA_TYPE ijgf[I_SIZE][J_SIZE][G_SIZE][F_SIZE], 
                                          DATA_TYPE klhj[K_SIZE][L_SIZE][H_SIZE][J_SIZE], 
                                          DATA_TYPE i[I_SIZE], 
                                          DATA_TYPE k[K_SIZE], 
                                          DATA_TYPE l[L_SIZE], 
                                          DATA_TYPE *result) {
  int ai, bi, ci, di, ei, fi, gi, hi, ii, ji, ki, li;
  
  *result = 0.0f;
  
  // Actual computation for quantum circuit tensor network
  for (ai = 0; ai < A_SIZE; ai++) {
    for (bi = 0; bi < B_SIZE; bi++) {
      for (ci = 0; ci < C_SIZE; ci++) {
        for (di = 0; di < D_SIZE; di++) {
          for (ei = 0; ei < E_SIZE; ei++) {
            for (fi = 0; fi < F_SIZE; fi++) {
              for (gi = 0; gi < G_SIZE; gi++) {
                for (hi = 0; hi < H_SIZE; hi++) {
                  for (ii = 0; ii < I_SIZE; ii++) {
                    for (ji = 0; ji < J_SIZE; ji++) {
                      for (ki = 0; ki < K_SIZE; ki++) {
                        for (li = 0; li < L_SIZE; li++) {
                          *result += a[ai] * b[bi] * c[ci] * da[di][ai] * eb[ei][bi] * 
                                     fc[fi][ci] * ghde[gi][hi][di][ei] * ijgf[ii][ji][gi][fi] * 
                                     klhj[ki][li][hi][ji] * i[ii] * k[ki] * l[li];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
