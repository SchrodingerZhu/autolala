// Deriche Loop Nest 2: Backward pass in i-direction
#include <math.h>

#define W 720
#define H 480
#define DATA_TYPE float

volatile DATA_TYPE imgIn[733][488]; // W=720 padded to 733 (prime), H=480 padded
                                    // to 488 (8×61, 61 is prime)
volatile DATA_TYPE y2[751][568]; // W=720 padded to 751 (prime), H=480 padded to
                                 // 568 (8×71, 71 is prime)

void kernel_deriche2() {
  int i, j;
  DATA_TYPE xp1, xp2, yp1, yp2;
  DATA_TYPE a3, a4, b1, b2;

  for (i = 0; i < W; i++) {
    yp1 = 0.0f;
    yp2 = 0.0f;
    xp1 = 0.0f;
    xp2 = 0.0f;
    for (j = H - 1; j >= 0; j--) {
      y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
      xp2 = xp1;
      xp1 = imgIn[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }
}
