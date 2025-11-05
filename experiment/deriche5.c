// Deriche Loop Nest 5: Backward pass in j-direction
#include <math.h>

#define W 720
#define H 480
#define DATA_TYPE float

volatile DATA_TYPE imgOut[739][584]; // W=720 padded to 739 (prime), H=480
                                     // padded to 584 (8×73, 73 is prime)
volatile DATA_TYPE y2[751][568]; // W=720 padded to 751 (prime), H=480 padded to
                                 // 568 (8×71, 71 is prime)

void kernel_deriche5() {
  int i, j;
  DATA_TYPE tp1, tp2, yp1, yp2;
  DATA_TYPE a3, a4, b1, b2;

  for (j = 0; j < H; j++) {
    tp1 = 0.0f;
    tp2 = 0.0f;
    yp1 = 0.0f;
    yp2 = 0.0f;
    for (i = W - 1; i >= 0; i--) {
      y2[i][j] = a3 * tp1 + a4 * tp2 + b1 * yp1 + b2 * yp2;
      tp2 = tp1;
      tp1 = imgOut[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }
}
