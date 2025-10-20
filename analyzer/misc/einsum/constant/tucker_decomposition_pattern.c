#define DATA_TYPE float
#define I_SIZE 64
#define J_SIZE 64
#define K_SIZE 64
#define L_SIZE 64
#define A_DIM 64
#define B_DIM 64
#define C_DIM 64
#define D_DIM 64

// Optimized Tucker decomposition: stepwise mode contractions
void kernel_tucker_decomposition_pattern_opt(
    const DATA_TYPE X[I_SIZE][J_SIZE][K_SIZE][72],  // L_SIZE=64 padded to 72
    const DATA_TYPE A[A_DIM][72],  // I_SIZE=64 padded to 72
    const DATA_TYPE B[B_DIM][72],  // J_SIZE=64 padded to 72
    const DATA_TYPE C[C_DIM][72],  // K_SIZE=64 padded to 72
    const DATA_TYPE D[D_DIM][72],  // L_SIZE=64 padded to 72
    DATA_TYPE tmp1[A_DIM][J_SIZE][K_SIZE][72],  // L_SIZE=64 padded to 72
    DATA_TYPE tmp2[A_DIM][B_DIM][K_SIZE][72],  // L_SIZE=64 padded to 72
    DATA_TYPE tmp3[A_DIM][B_DIM][C_DIM][72],  // L_SIZE=64 padded to 72
    DATA_TYPE Y[A_DIM][B_DIM][C_DIM][72])  // D_DIM=64 padded to 72
{
    int a,b,c,d,i,j,k,l;

    // --- Step 1: contract mode-i with A ---
    for (a=0;a<A_DIM;a++)
        for (j=0;j<J_SIZE;j++)
            for (k=0;k<K_SIZE;k++)
                for (l=0;l<L_SIZE;l++) {
                    DATA_TYPE s = 0;
                    for (i=0;i<I_SIZE;i++)
                        s += A[a][i] * X[i][j][k][l];
                    tmp1[a][j][k][l] = s;
                }

    // --- Step 2: contract mode-j with B ---
    for (a=0;a<A_DIM;a++)
        for (b=0;b<B_DIM;b++)
            for (k=0;k<K_SIZE;k++)
                for (l=0;l<L_SIZE;l++) {
                    DATA_TYPE s = 0;
                    for (j=0;j<J_SIZE;j++)
                        s += B[b][j] * tmp1[a][j][k][l];
                    tmp2[a][b][k][l] = s;
                }

    // --- Step 3: contract mode-k with C ---
    for (a=0;a<A_DIM;a++)
        for (b=0;b<B_DIM;b++)
            for (c=0;c<C_DIM;c++)
                for (l=0;l<L_SIZE;l++) {
                    DATA_TYPE s = 0;
                    for (k=0;k<K_SIZE;k++)
                        s += C[c][k] * tmp2[a][b][k][l];
                    tmp3[a][b][c][l] = s;
                }

    // --- Step 4: contract mode-l with D ---
    for (a=0;a<A_DIM;a++)
        for (b=0;b<B_DIM;b++)
            for (c=0;c<C_DIM;c++)
                for (d=0;d<D_DIM;d++) {
                    DATA_TYPE s = 0;
                    for (l=0;l<L_SIZE;l++)
                        s += D[d][l] * tmp3[a][b][c][l];
                    Y[a][b][c][d] = s;
                }
}
