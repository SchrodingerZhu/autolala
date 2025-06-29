module {
  func.func @test(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, 
                   %M: index, %N: index, %K: index) {  // Removed %arg6, %arg7, %arg8
    affine.for %i = 0 to %M {  // Shifted from %arg9 to %arg6
      affine.for %j = 0 to %N {  // Shifted from %arg10 to %arg7
        %m = affine.load %arg1[%i * 16 + %j] : memref<?xf64>
        affine.for %k = 0 to %K {  // Shifted from %arg11 to %arg8
          affine.for %kk = 0 to 16 {  // Shifted from %arg12 to %arg9
                %0 = affine.load %arg0[%i * 16 + %kk + %j * 16] : memref<?xf64>
                affine.store %0, %arg2[%k * 16 + %kk] : memref<?xf64>
            }
          }
        affine.for %k = 0 to %K {
            affine.store %m, %arg2[%i * 16 + %k] : memref<?xf64>
        }
      }
    }
    func.return
  }
}
