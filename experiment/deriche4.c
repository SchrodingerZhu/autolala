// Deriche Loop Nest 4: Forward pass in j-direction
#include <math.h>

#define W 720
#define H 480
#define DATA_TYPE float

volatile DATA_TYPE imgOut[739][584]; // W=720 padded to 739 (prime), H=480
                                     // padded to 584 (8×73, 73 is prime)
volatile DATA_TYPE Y1[743][536]; // W=720 padded to 743 (prime), H=480 padded to
                                 // 536 (8×67, 67 is prime)

void kernel_deriche4() {
  int i, j;
  DATA_TYPE tm1, ym1, ym2;
  DATA_TYPE a1, a2, b1, b2;

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
}
