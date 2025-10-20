#define DATA_TYPE float
#define B_SIZE 64
#define C_SIZE 64
#define D_SIZE 64
#define E_SIZE 64
#define F_SIZE 64
#define G_SIZE 64

// Optimized weighted model counting kernel
void kernel_weighted_model_counting_pattern_opt(
    const DATA_TYPE b[72],  // B_SIZE=64 padded to 72
    const DATA_TYPE c[72],  // C_SIZE=64 padded to 72
    const DATA_TYPE d[72],  // D_SIZE=64 padded to 72
    const DATA_TYPE e[72],  // E_SIZE=64 padded to 72
    const DATA_TYPE f[72],  // F_SIZE=64 padded to 72
    const DATA_TYPE ef[E_SIZE][72],  // F_SIZE=64 padded to 72
    const DATA_TYPE eg[E_SIZE][72],  // G_SIZE=64 padded to 72
    const DATA_TYPE bc[B_SIZE][72],  // C_SIZE=64 padded to 72
    const DATA_TYPE cdc[C_SIZE][D_SIZE][72],  // C_SIZE=64 padded to 72
    DATA_TYPE tmp_bc[B_SIZE][72],  // C_SIZE=64 padded to 72
    DATA_TYPE tmp_cd[C_SIZE][72],  // D_SIZE=64 padded to 72
    DATA_TYPE tmp_ef[E_SIZE][72],  // F_SIZE=64 padded to 72
    DATA_TYPE tmp_eg[E_SIZE][72],  // G_SIZE=64 padded to 72
    DATA_TYPE *result)
{
    int bi, ci, di, ei, fi, gi;

    // --- 1. Combine unary b,c with pair bc
    for (bi = 0; bi < B_SIZE; ++bi)
        for (ci = 0; ci < C_SIZE; ++ci)
            tmp_bc[bi][ci] = b[bi] * c[ci] * bc[bi][ci];

    // --- 2. Precompute c–d correlations
    for (ci = 0; ci < C_SIZE; ++ci)
        for (di = 0; di < D_SIZE; ++di) {
            DATA_TYPE s = 0;
            for (int c2 = 0; c2 < C_SIZE; ++c2)
                s += cdc[ci][di][c2] * c[c2];
            tmp_cd[ci][di] = s * d[di];
        }

    // --- 3. Combine e,f,g with ef,eg
    for (ei = 0; ei < E_SIZE; ++ei)
        for (fi = 0; fi < F_SIZE; ++fi)
            tmp_ef[ei][fi] = e[ei] * f[fi] * ef[ei][fi];

    for (ei = 0; ei < E_SIZE; ++ei)
        for (gi = 0; gi < G_SIZE; ++gi)
            tmp_eg[ei][gi] = e[ei] * eg[ei][gi];

    // --- 4. Accumulate partial sums
    DATA_TYPE sum_bc_cd = 0.0f;
    for (bi = 0; bi < B_SIZE; ++bi)
        for (ci = 0; ci < C_SIZE; ++ci)
            for (di = 0; di < D_SIZE; ++di)
                sum_bc_cd += tmp_bc[bi][ci] * tmp_cd[ci][di];

    DATA_TYPE sum_ef_eg = 0.0f;
    for (ei = 0; ei < E_SIZE; ++ei) {
        DATA_TYPE s1 = 0, s2 = 0;
        for (fi = 0; fi < F_SIZE; ++fi)
            s1 += tmp_ef[ei][fi];
        for (gi = 0; gi < G_SIZE; ++gi)
            s2 += tmp_eg[ei][gi];
        sum_ef_eg += s1 * s2;
    }

    // --- 5. Final contraction
    *result = sum_bc_cd * sum_ef_eg;
}
