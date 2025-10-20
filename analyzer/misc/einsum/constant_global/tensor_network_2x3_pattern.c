#define DATA_TYPE double
#define I_SIZE 64
#define J_SIZE 64
#define M_SIZE 64
#define L_SIZE 64
#define K_SIZE 64
#define N_SIZE 64
#define O_SIZE 64

extern volatile DATA_TYPE A[I_SIZE][J_SIZE];
extern volatile DATA_TYPE B[I_SIZE][M_SIZE][L_SIZE];
extern volatile DATA_TYPE C[L_SIZE][O_SIZE];
extern volatile DATA_TYPE D[J_SIZE][K_SIZE];
extern volatile DATA_TYPE E[K_SIZE][M_SIZE][N_SIZE];
extern volatile DATA_TYPE F[N_SIZE][O_SIZE];
extern volatile DATA_TYPE CF[N_SIZE][L_SIZE];
extern volatile DATA_TYPE ECF[K_SIZE][M_SIZE][L_SIZE];
extern volatile DATA_TYPE DECF[J_SIZE][M_SIZE][L_SIZE];
extern volatile DATA_TYPE ADECF[I_SIZE][M_SIZE][L_SIZE];
extern volatile DATA_TYPE BMerged[M_SIZE][L_SIZE];
extern volatile DATA_TYPE result[1];

// temporaries passed by caller
void kernel_tensor_network_2x3_pattern_opt()
{
    int i,j,m,l,k,n,o;

    // --- Step 1: CF[n,l] = Σ_o F[n,o]*C[l,o]
    for (n=0;n<N_SIZE;n++)
        for (l=0;l<L_SIZE;l++) {
            DATA_TYPE s=0;
            for (o=0;o<O_SIZE;o++)
                s += (double)F[n][o] * (double)C[l][o];
            CF[n][l]=s;
        }

    // --- Step 2: ECF[k,m,l] = Σ_n E[k,m,n]*CF[n,l]
    for (k=0;k<K_SIZE;k++)
        for (m=0;m<M_SIZE;m++)
            for (l=0;l<L_SIZE;l++) {
                DATA_TYPE s=0;
                for (n=0;n<N_SIZE;n++)
                    s += (double)E[k][m][n]*CF[n][l];
                ECF[k][m][l]=s;
            }

    // --- Step 3: DECF[j,m,l] = Σ_k D[j,k]*ECF[k,m,l]
    for (j=0;j<J_SIZE;j++)
        for (m=0;m<M_SIZE;m++)
            for (l=0;l<L_SIZE;l++) {
                DATA_TYPE s=0;
                for (k=0;k<K_SIZE;k++)
                    s += (double)D[j][k]*ECF[k][m][l];
                DECF[j][m][l]=s;
            }

    // --- Step 4: ADECF[i,m,l] = Σ_j A[i][j]*DECF[j,m,l]
    for (i=0;i<I_SIZE;i++)
        for (m=0;m<M_SIZE;m++)
            for (l=0;l<L_SIZE;l++) {
                DATA_TYPE s=0;
                for (j=0;j<J_SIZE;j++)
                    s += (double)A[i][j]*DECF[j][m][l];
                ADECF[i][m][l]=s;
            }

    // --- Step 5: BMerged[m,l] = Σ_i B[i,m,l]*ADECF[i,m,l]
    for (m=0;m<M_SIZE;m++)
        for (l=0;l<L_SIZE;l++) {
            DATA_TYPE s=0;
            for (i=0;i<I_SIZE;i++)
                s += (double)B[i][m][l]*ADECF[i][m][l];
            BMerged[m][l]=s;
        }

    // --- Step 6: final scalar
    double total=0;
    for (m=0;m<M_SIZE;m++)
        for (l=0;l<L_SIZE;l++)
            total += BMerged[m][l];
    result[0]=total;
}
