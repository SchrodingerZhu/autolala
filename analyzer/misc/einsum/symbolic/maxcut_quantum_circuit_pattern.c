#define DATA_TYPE float
#define LIMIT 1024
#define TINY_LIMIT 4   // Use extremely small dimensions for highly complex patterns
typedef __SIZE_TYPE__ size_t;

// Max-Cut quantum circuit: a,b,c,da,eb,fc,ghde,ijgf,klhj,i,k,l->
// Memory access pattern: quantum circuit tensor network contraction
// Contract tensors step by step; all arrays passed in are preallocated.
void kernel_maxcut_quantum_circuit_pattern(size_t A, size_t B, size_t C, size_t D, size_t E, 
                                          size_t F, size_t G, size_t H, size_t I, size_t J, 
                                          size_t K, size_t L,
                                          DATA_TYPE a[TINY_LIMIT], 
                                          DATA_TYPE b[TINY_LIMIT], 
                                          DATA_TYPE c[TINY_LIMIT], 
                                          DATA_TYPE da[TINY_LIMIT][TINY_LIMIT], 
                                          DATA_TYPE eb[TINY_LIMIT][TINY_LIMIT], 
                                          DATA_TYPE fc[TINY_LIMIT][TINY_LIMIT], 
                                          DATA_TYPE ghde[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT], 
                                          DATA_TYPE ijgf[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT], 
                                          DATA_TYPE klhj[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT], 
                                          DATA_TYPE i[TINY_LIMIT], 
                                          DATA_TYPE k[TINY_LIMIT], 
                                          DATA_TYPE l[TINY_LIMIT], 
                                          DATA_TYPE tmp_da[TINY_LIMIT],
                                          DATA_TYPE tmp_eb[TINY_LIMIT],
                                          DATA_TYPE tmp_fc[TINY_LIMIT],
                                          DATA_TYPE tmp_gh[TINY_LIMIT][TINY_LIMIT],
                                          DATA_TYPE tmp_ij[TINY_LIMIT][TINY_LIMIT],
                                          DATA_TYPE tmp_kl[TINY_LIMIT][TINY_LIMIT],
                                          DATA_TYPE *result)
{
    // step 1: combine 1D + 2D
    for (int d = 0; d < D; ++d) {
        DATA_TYPE s = 0;
        for (int a_i = 0; a_i < A; ++a_i)
            s += a[a_i] * da[d][a_i];
        tmp_da[d] = s;
    }
    for (int e = 0; e < E; ++e) {
        DATA_TYPE s = 0;
        for (int b_i = 0; b_i < B; ++b_i)
            s += b[b_i] * eb[e][b_i];
        tmp_eb[e] = s;
    }
    for (int f = 0; f < F; ++f) {
        DATA_TYPE s = 0;
        for (int c_i = 0; c_i < C; ++c_i)
            s += c[c_i] * fc[f][c_i];
        tmp_fc[f] = s;
    }

    // step 2: contract ghde with reduced da, eb
    for (int g = 0; g < G; ++g)
        for (int h = 0; h < H; ++h) {
            DATA_TYPE s = 0;
            for (int d = 0; d < D; ++d)
                for (int e = 0; e < E; ++e)
                    s += ghde[g][h][d][e] * tmp_da[d] * tmp_eb[e];
            tmp_gh[g][h] = s;
        }

    // step 3: contract ijgf with tmp_gh and tmp_fc
    for (int ii = 0; ii < I; ++ii)
        for (int jj = 0; jj < J; ++jj) {
            DATA_TYPE s = 0;
            for (int g = 0; g < G; ++g)
                for (int f = 0; f < F; ++f)
                    s += ijgf[ii][jj][g][f] * tmp_gh[g][jj % H] * tmp_fc[f];
            tmp_ij[ii][jj] = s;
        }

    // step 4: contract klhj
    for (int kk = 0; kk < K; ++kk)
        for (int ll = 0; ll < L; ++ll) {
            DATA_TYPE s = 0;
            for (int h = 0; h < H; ++h)
                for (int j = 0; j < J; ++j)
                    s += klhj[kk][ll][h][j];
            tmp_kl[kk][ll] = s;
        }

    // step 5: final scalar contraction
    *result = 0;
    for (int ii = 0; ii < I; ++ii)
        for (int jj = 0; jj < J; ++jj)
            for (int kk = 0; kk < K; ++kk)
                for (int ll = 0; ll < L; ++ll)
                    *result += tmp_ij[ii][jj] * tmp_kl[kk][ll]
                              * i[ii] * k[kk] * l[ll];
}
