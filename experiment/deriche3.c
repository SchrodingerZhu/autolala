// Deriche Loop Nest 3: Combine Y1 and y2 into imgOut
#include <math.h>

#define W 720
#define H 480
#define DATA_TYPE float

volatile DATA_TYPE imgOut[739][584]; // W=720 padded to 739 (prime), H=480
                                     // padded to 584 (8×73, 73 is prime)
volatile DATA_TYPE Y1[743][536]; // W=720 padded to 743 (prime), H=480 padded to
                                 // 536 (8×67, 67 is prime)
volatile DATA_TYPE y2[751][568]; // W=720 padded to 751 (prime), H=480 padded to
                                 // 568 (8×71, 71 is prime)

void kernel_deriche3() {
  int i, j;
  DATA_TYPE c1;

  for (i = 0; i < W; i++)
    for (j = 0; j < H; j++) {
      imgOut[i][j] = c1 * (Y1[i][j] + y2[i][j]);
    }
}
