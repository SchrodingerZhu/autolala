#include <stddef.h>

#define DATA_TYPE float
#define LIMIT 1024
#define TINY_LIMIT 8   // Use very small dimensions for extremely complex patterns

// Weighted model counting: b,c,d,e,f,ef,eg,bc,cdc->
// Memory access pattern: complex boolean/probabilistic tensor contraction
void kernel_weighted_model_counting_pattern(size_t B, size_t C, size_t D, size_t E, 
                                           size_t F, size_t G,
                                           DATA_TYPE b[TINY_LIMIT], 
                                           DATA_TYPE c[TINY_LIMIT], 
                                           DATA_TYPE d[TINY_LIMIT], 
                                           DATA_TYPE e[TINY_LIMIT], 
                                           DATA_TYPE f[TINY_LIMIT], 
                                           DATA_TYPE ef[TINY_LIMIT][TINY_LIMIT], 
                                           DATA_TYPE eg[TINY_LIMIT][TINY_LIMIT], 
                                           DATA_TYPE bc[TINY_LIMIT][TINY_LIMIT], 
                                           DATA_TYPE cdc[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT], 
                                           DATA_TYPE *result) {
  int bi, ci, di, ei, fi, gi;
  
  *result = 0;
  
  // Access pattern for weighted model counting over all variable assignments
  for (bi = 0; bi < B; bi++) {
    b[bi] = 0; // Access b[bi]
    for (ci = 0; ci < C; ci++) {
      c[ci] = 0; // Access c[ci]
      bc[bi][ci] = 0; // Access bc[bi][ci]
      for (di = 0; di < D; di++) {
        d[di] = 0; // Access d[di]
        cdc[ci][di][ci] = 0; // Access cdc[ci][di][ci]
        for (ei = 0; ei < E; ei++) {
          e[ei] = 0; // Access e[ei]
          for (fi = 0; fi < F; fi++) {
            f[fi] = 0; // Access f[fi]
            ef[ei][fi] = 0; // Access ef[ei][fi]
            for (gi = 0; gi < G; gi++) {
              eg[ei][gi] = 0; // Access eg[ei][gi]
            }
          }
        }
      }
    }
  }
}