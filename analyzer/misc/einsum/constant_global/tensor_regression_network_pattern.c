#define DATA_TYPE double
#define TINY_SIZE 16
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



volatile DATA_TYPE X[A_SIZE][B_SIZE][C_SIZE][D_SIZE][24];
volatile DATA_TYPE Y[F_SIZE][G_SIZE][H_SIZE][I_SIZE][24];
volatile DATA_TYPE M1[B_SIZE][24];
volatile DATA_TYPE M2[C_SIZE][24];
volatile DATA_TYPE M3[D_SIZE][24];
volatile DATA_TYPE M4[E_SIZE][24];
volatile DATA_TYPE M5[K_SIZE][24];
volatile DATA_TYPE tmp1[A_SIZE][F_SIZE][C_SIZE][D_SIZE][24];
volatile DATA_TYPE tmp2[A_SIZE][F_SIZE][G_SIZE][D_SIZE][24];
volatile DATA_TYPE tmp3[A_SIZE][F_SIZE][G_SIZE][H_SIZE][24];
volatile DATA_TYPE tmp4[A_SIZE][F_SIZE][G_SIZE][H_SIZE][24];
volatile DATA_TYPE Y1[F_SIZE][G_SIZE][H_SIZE][I_SIZE][24];
volatile DATA_TYPE result[A_SIZE][24];
// Optimized tensor regression network contraction: stepwise version
void kernel_tensor_regression_network_pattern_opt() {
    int a,b,c,d,e,f,g,h,i,j,k;

    // Initialize result
    for (a=0;a<A_SIZE;a++)
        for (k=0;k<K_SIZE;k++)
            result[a][k]=0;

    // --- Contract M1[b,f] ---
    for (a=0;a<A_SIZE;a++)
        for (f=0;f<F_SIZE;f++)
            for (c=0;c<C_SIZE;c++)
                for (d=0;d<D_SIZE;d++)
                    for (e=0;e<E_SIZE;e++) {
                        DATA_TYPE s=0;
                        for (b=0;b<B_SIZE;b++)
                            s+=X[a][b][c][d][e]*M1[b][f];
                        tmp1[a][f][c][d][e]=s;
                    }

    // --- Contract M2[c,g] ---
    for (a=0;a<A_SIZE;a++)
        for (f=0;f<F_SIZE;f++)
            for (g=0;g<G_SIZE;g++)
                for (d=0;d<D_SIZE;d++)
                    for (e=0;e<E_SIZE;e++) {
                        DATA_TYPE s=0;
                        for (c=0;c<C_SIZE;c++)
                            s+=tmp1[a][f][c][d][e]*M2[c][g];
                        tmp2[a][f][g][d][e]=s;
                    }

    // --- Contract M3[d,h] ---
    for (a=0;a<A_SIZE;a++)
        for (f=0;f<F_SIZE;f++)
            for (g=0;g<G_SIZE;g++)
                for (h=0;h<H_SIZE;h++)
                    for (e=0;e<E_SIZE;e++) {
                        DATA_TYPE s=0;
                        for (d=0;d<D_SIZE;d++)
                            s+=tmp2[a][f][g][d][e]*M3[d][h];
                        tmp3[a][f][g][h][e]=s;
                    }

    // --- Contract M4[e,i] ---
    for (a=0;a<A_SIZE;a++)
        for (f=0;f<F_SIZE;f++)
            for (g=0;g<G_SIZE;g++)
                for (h=0;h<H_SIZE;h++)
                    for (i=0;i<I_SIZE;i++) {
                        DATA_TYPE s=0;
                        for (e=0;e<E_SIZE;e++)
                            s+=tmp3[a][f][g][h][e]*M4[e][i];
                        tmp4[a][f][g][h][i]=s;
                    }

    // --- Pre-contract Y with M5[k,j] ---
    for (f=0;f<F_SIZE;f++)
        for (g=0;g<G_SIZE;g++)
            for (h=0;h<H_SIZE;h++)
                for (i=0;i<I_SIZE;i++)
                    for (k=0;k<K_SIZE;k++) {
                        DATA_TYPE s=0;
                        for (j=0;j<J_SIZE;j++)
                            s+=Y[f][g][h][i][j]*M5[k][j];
                        Y1[f][g][h][i][k]=s;
                    }

    // --- Final contraction over f,g,h,i ---
    for (a=0;a<A_SIZE;a++)
        for (k=0;k<K_SIZE;k++) {
            DATA_TYPE s=0;
            for (f=0;f<F_SIZE;f++)
                for (g=0;g<G_SIZE;g++)
                    for (h=0;h<H_SIZE;h++)
                        for (i=0;i<I_SIZE;i++)
                            s+=tmp4[a][f][g][h][i]*Y1[f][g][h][i][k];
            result[a][k]=s;
        }
}
