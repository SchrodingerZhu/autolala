// Deriche Loop Nest 1: Forward pass in i-direction
#include <math.h>

#define W 720
#define H 480
#define DATA_TYPE float

volatile DATA_TYPE imgIn[733][488]; // W=720 padded to 733 (prime), H=480 padded
                                    // to 488 (8×61, 61 is prime)
volatile DATA_TYPE Y1[743][536]; // W=720 padded to 743 (prime), H=480 padded to
                                 // 536 (8×67, 67 is prime)

void kernel_deriche1() {
  int i, j;
  DATA_TYPE xm1, ym1, ym2;
  DATA_TYPE a1, a2, b1, b2;

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
}
