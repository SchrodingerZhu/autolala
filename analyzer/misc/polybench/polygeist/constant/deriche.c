// Configuration from: https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/blob/master/medley/deriche/deriche.h
#include <math.h>

#define W 720
#define H 480
#define DATA_TYPE float


volatile DATA_TYPE imgIn[W][480];  // H=480 already multiple of 12
volatile DATA_TYPE imgOut[W][480];  // H=480 already multiple of 12
volatile DATA_TYPE Y1[W][480];  // H=480 already multiple of 12
volatile DATA_TYPE y2[W][480];  // H=480 already multiple of 12

void kernel_deriche(DATA_TYPE alpha) {
  int i, j;
  DATA_TYPE xm1, tm1, ym1, ym2;
  DATA_TYPE xp1, xp2;
  DATA_TYPE tp1, tp2;
  DATA_TYPE yp1, yp2;

  DATA_TYPE k = (1.0f - expf(-alpha)) * (1.0f - expf(-alpha)) / (1.0f + 2.0f * alpha * expf(-alpha) - expf(2.0f * alpha));
  DATA_TYPE a1 = k;
  DATA_TYPE a2 = k * expf(-alpha) * (alpha - 1.0f);
  DATA_TYPE a3 = k * expf(-alpha) * (alpha + 1.0f);
  DATA_TYPE a4 = -k * expf(-2.0f * alpha);
  DATA_TYPE b1 = powf(2.0f, -alpha);
  DATA_TYPE b2 = -expf(-2.0f * alpha);
  DATA_TYPE c1 = 1;
  DATA_TYPE c2 = 1;

  for (i = 0; i < W; i++) {
    ym1 = 0.0f;
    ym2 = 0.0f;
    xm1 = 0.0f;
    for (j = 0; j < H; j++) {
      Y1[i][j] = a1 * imgIn[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
      xm1 = imgIn[i][j];
      ym2 = ym1;
      ym1 = Y1[i][j];
    }
  }

  for (i = 0; i < W; i++) {
    yp1 = 0.0f;
    yp2 = 0.0f;
    xp1 = 0.0f;
    xp2 = 0.0f;
    for (j = H-1; j >= 0; j--) {
      y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
      xp2 = xp1;
      xp1 = imgIn[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }

  for (i = 0; i < W; i++)
    for (j = 0; j < H; j++) {
      imgOut[i][j] = c1 * (Y1[i][j] + y2[i][j]);
    }

  for (j = 0; j < H; j++) {
    tm1 = 0.0f;
    ym1 = 0.0f;
    ym2 = 0.0f;
    for (i = 0; i < W; i++) {
      Y1[i][j] = a1 * imgOut[i][j] + a2 * tm1 + b1 * ym1 + b2 * ym2;
      tm1 = imgOut[i][j];
      ym2 = ym1;
      ym1 = Y1[i][j];
    }
  }

  for (j = 0; j < H; j++) {
    tp1 = 0.0f;
    tp2 = 0.0f;
    yp1 = 0.0f;
    yp2 = 0.0f;
    for (i = W-1; i >= 0; i--) {
      y2[i][j] = a3 * tp1 + a4 * tp2 + b1 * yp1 + b2 * yp2;
      tp2 = tp1;
      tp1 = imgOut[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }

  for (i = 0; i < W; i++)
    for (j = 0; j < H; j++)
      imgOut[i][j] = c2 * (Y1[i][j] + y2[i][j]);
}
