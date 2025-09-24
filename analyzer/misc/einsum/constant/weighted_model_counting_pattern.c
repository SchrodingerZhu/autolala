#define DATA_TYPE float
#define B_SIZE 4
#define C_SIZE 4
#define D_SIZE 4
#define E_SIZE 4
#define F_SIZE 4
#define G_SIZE 4

// Weighted model counting: b,c,d,e,f,ef,eg,bc,cdc->
// Memory access pattern: complex boolean/probabilistic tensor contraction
void kernel_weighted_model_counting_pattern(DATA_TYPE b[B_SIZE], 
                                           DATA_TYPE c[C_SIZE], 
                                           DATA_TYPE d[D_SIZE], 
                                           DATA_TYPE e[E_SIZE], 
                                           DATA_TYPE f[F_SIZE], 
                                           DATA_TYPE ef[E_SIZE][F_SIZE], 
                                           DATA_TYPE eg[E_SIZE][G_SIZE], 
                                           DATA_TYPE bc[B_SIZE][C_SIZE], 
                                           DATA_TYPE cdc[C_SIZE][D_SIZE][C_SIZE], 
                                           DATA_TYPE *result) {
  int bi, ci, di, ei, fi, gi;
  
  *result = 0.0f;
  
  // Actual computation for weighted model counting over all variable assignments
  for (bi = 0; bi < B_SIZE; bi++) {
    for (ci = 0; ci < C_SIZE; ci++) {
      for (di = 0; di < D_SIZE; di++) {
        for (ei = 0; ei < E_SIZE; ei++) {
          for (fi = 0; fi < F_SIZE; fi++) {
            for (gi = 0; gi < G_SIZE; gi++) {
              *result += b[bi] * c[ci] * d[di] * e[ei] * f[fi] * 
                         ef[ei][fi] * eg[ei][gi] * bc[bi][ci] * cdc[ci][di][ci];
            }
          }
        }
      }
    }
  }
}
