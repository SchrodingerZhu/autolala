#define DATA_TYPE float
#define SIZE 64

// Contract tensors step by step; all arrays passed in are preallocated.
void kernel_maxcut_quantum_circuit_pattern(
    const DATA_TYPE a[72],  // SIZE=64 padded to 72
    const DATA_TYPE b[72],  // SIZE=64 padded to 72
    const DATA_TYPE c[72],  // SIZE=64 padded to 72
    const DATA_TYPE da[SIZE][72],  // SIZE=64 padded to 72
    const DATA_TYPE eb[SIZE][72],  // SIZE=64 padded to 72
    const DATA_TYPE fc[SIZE][72],  // SIZE=64 padded to 72
    const DATA_TYPE ghde[SIZE][SIZE][SIZE][72],  // SIZE=64 padded to 72
    const DATA_TYPE ijgf[SIZE][SIZE][SIZE][72],  // SIZE=64 padded to 72
    const DATA_TYPE klhj[SIZE][SIZE][SIZE][72],  // SIZE=64 padded to 72
    const DATA_TYPE i_vec[72],  // SIZE=64 padded to 72
    const DATA_TYPE k_vec[72],  // SIZE=64 padded to 72
    const DATA_TYPE l_vec[72],  // SIZE=64 padded to 72
    DATA_TYPE tmp_da[72],  // SIZE=64 padded to 72
    DATA_TYPE tmp_eb[72],  // SIZE=64 padded to 72
    DATA_TYPE tmp_fc[72],  // SIZE=64 padded to 72
    DATA_TYPE tmp_gh[SIZE][72],  // SIZE=64 padded to 72
    DATA_TYPE tmp_ij[SIZE][72],  // SIZE=64 padded to 72
    DATA_TYPE tmp_kl[SIZE][72],  // SIZE=64 padded to 72
    DATA_TYPE *result)
{
    // step 1: combine 1D + 2D
    for (int d = 0; d < SIZE; ++d) {
        DATA_TYPE s = 0;
        for (int a_i = 0; a_i < SIZE; ++a_i)
            s += a[a_i] * da[d][a_i];
        tmp_da[d] = s;
    }
    for (int e = 0; e < SIZE; ++e) {
        DATA_TYPE s = 0;
        for (int b_i = 0; b_i < SIZE; ++b_i)
            s += b[b_i] * eb[e][b_i];
        tmp_eb[e] = s;
    }
    for (int f = 0; f < SIZE; ++f) {
        DATA_TYPE s = 0;
        for (int c_i = 0; c_i < SIZE; ++c_i)
            s += c[c_i] * fc[f][c_i];
        tmp_fc[f] = s;
    }

    // step 2: contract ghde with reduced da, eb
    for (int g = 0; g < SIZE; ++g)
        for (int h = 0; h < SIZE; ++h) {
            DATA_TYPE s = 0;
            for (int d = 0; d < SIZE; ++d)
                for (int e = 0; e < SIZE; ++e)
                    s += ghde[g][h][d][e] * tmp_da[d] * tmp_eb[e];
            tmp_gh[g][h] = s;
        }

    // step 3: contract ijgf with tmp_gh and tmp_fc
    for (int ii = 0; ii < SIZE; ++ii)
        for (int jj = 0; jj < SIZE; ++jj) {
            DATA_TYPE s = 0;
            for (int g = 0; g < SIZE; ++g)
                for (int f = 0; f < SIZE; ++f)
                    s += ijgf[ii][jj][g][f] * tmp_gh[g][jj % SIZE] * tmp_fc[f];
            tmp_ij[ii][jj] = s;
        }

    // step 4: contract klhj
    for (int kk = 0; kk < SIZE; ++kk)
        for (int ll = 0; ll < SIZE; ++ll) {
            DATA_TYPE s = 0;
            for (int h = 0; h < SIZE; ++h)
                for (int j = 0; j < SIZE; ++j)
                    s += klhj[kk][ll][h][j];
            tmp_kl[kk][ll] = s;
        }

    // step 5: final scalar contraction
    *result = 0;
    for (int ii = 0; ii < SIZE; ++ii)
        for (int jj = 0; jj < SIZE; ++jj)
            for (int kk = 0; kk < SIZE; ++kk)
                for (int ll = 0; ll < SIZE; ++ll)
                    *result += tmp_ij[ii][jj] * tmp_kl[kk][ll]
                              * i_vec[ii] * k_vec[kk] * l_vec[ll];
}
