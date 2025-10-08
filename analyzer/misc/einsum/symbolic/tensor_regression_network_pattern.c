#define DATA_TYPE float
#define LIMIT 1024
#define TINY_LIMIT 8
typedef __SIZE_TYPE__ size_t;

// Tensor regression network: abcde,fghij,bf,cg,dh,ei,kj->ak
// Memory access pattern: complex tensor network with multiple contractions
// Optimized tensor regression network contraction: stepwise version
void kernel_tensor_regression_network_pattern(size_t A, size_t B, size_t C, size_t D, size_t E,
                                             size_t F, size_t G, size_t H, size_t I, size_t J, size_t K,
                                             DATA_TYPE X[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT], // abcde
                                             DATA_TYPE Y[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT], // fghij  
                                             DATA_TYPE M1[TINY_LIMIT][TINY_LIMIT], // bf
                                             DATA_TYPE M2[TINY_LIMIT][TINY_LIMIT], // cg
                                             DATA_TYPE M3[TINY_LIMIT][TINY_LIMIT], // dh
                                             DATA_TYPE M4[TINY_LIMIT][TINY_LIMIT], // ei
                                             DATA_TYPE M5[TINY_LIMIT][TINY_LIMIT], // kj
                                             DATA_TYPE tmp1[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT],
                                             DATA_TYPE tmp2[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT],
                                             DATA_TYPE tmp3[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT],
                                             DATA_TYPE tmp4[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT],
                                             DATA_TYPE Y1[TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT][TINY_LIMIT],
                                             DATA_TYPE result[TINY_LIMIT][TINY_LIMIT]) // ak
{
    int a,b,c,d,e,f,g,h,i,j,k;

    // Initialize result
    for (a=0;a<A;a++)
        for (k=0;k<K;k++)
            result[a][k]=0;

    // --- Contract M1[b,f] ---
    for (a=0;a<A;a++)
        for (f=0;f<F;f++)
            for (c=0;c<C;c++)
                for (d=0;d<D;d++)
                    for (e=0;e<E;e++) {
                        DATA_TYPE s=0;
                        for (b=0;b<B;b++)
                            s+=X[a][b][c][d][e]*M1[b][f];
                        tmp1[a][f][c][d][e]=s;
                    }

    // --- Contract M2[c,g] ---
    for (a=0;a<A;a++)
        for (f=0;f<F;f++)
            for (g=0;g<G;g++)
                for (d=0;d<D;d++)
                    for (e=0;e<E;e++) {
                        DATA_TYPE s=0;
                        for (c=0;c<C;c++)
                            s+=tmp1[a][f][c][d][e]*M2[c][g];
                        tmp2[a][f][g][d][e]=s;
                    }

    // --- Contract M3[d,h] ---
    for (a=0;a<A;a++)
        for (f=0;f<F;f++)
            for (g=0;g<G;g++)
                for (h=0;h<H;h++)
                    for (e=0;e<E;e++) {
                        DATA_TYPE s=0;
                        for (d=0;d<D;d++)
                            s+=tmp2[a][f][g][d][e]*M3[d][h];
                        tmp3[a][f][g][h][e]=s;
                    }

    // --- Contract M4[e,i] ---
    for (a=0;a<A;a++)
        for (f=0;f<F;f++)
            for (g=0;g<G;g++)
                for (h=0;h<H;h++)
                    for (i=0;i<I;i++) {
                        DATA_TYPE s=0;
                        for (e=0;e<E;e++)
                            s+=tmp3[a][f][g][h][e]*M4[e][i];
                        tmp4[a][f][g][h][i]=s;
                    }

    // --- Pre-contract Y with M5[k,j] ---
    for (f=0;f<F;f++)
        for (g=0;g<G;g++)
            for (h=0;h<H;h++)
                for (i=0;i<I;i++)
                    for (k=0;k<K;k++) {
                        DATA_TYPE s=0;
                        for (j=0;j<J;j++)
                            s+=Y[f][g][h][i][j]*M5[k][j];
                        Y1[f][g][h][i][k]=s;
                    }

    // --- Final contraction over f,g,h,i ---
    for (a=0;a<A;a++)
        for (k=0;k<K;k++) {
            DATA_TYPE s=0;
            for (f=0;f<F;f++)
                for (g=0;g<G;g++)
                    for (h=0;h<H;h++)
                        for (i=0;i<I;i++)
                            s+=tmp4[a][f][g][h][i]*Y1[f][g][h][i][k];
            result[a][k]=s;
        }
}
