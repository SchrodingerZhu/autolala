module {
  func.func @doitgen(%A: memref<?x?x?xf64>, %C4: memref<?x?xf64>, %sum: memref<?xf64>, %NR : index, %NQ : index, %NP : index) {
    affine.for %r = 0 to %NR {
      // Loop over q
      affine.for %q = 0 to %NQ {
        // Initialize sum[p] = 0 for all p
        affine.for %p = 0 to %NP {
          %zero = arith.constant 0.0 : f64
          affine.store %zero, %sum[%p] : memref<?xf64>
        }
        // Tiled accumulation: sum[p] += A[r][q][s] * C4[s][p]
        // Tile both the contraction dim s and the output dim p so that a
        // 32x32 block of C4 (and the 32-wide sum tile + A row tile) stay
        // resident in cache and are reused across the tile.
        affine.for %pp = 0 to %NP step 32 {
          affine.for %ss = 0 to %NP step 32 {
            affine.for %p = 0 to 32 {
              affine.for %s = 0 to 32 {
                %A_val = affine.load %A[%r, %q, %ss + %s] : memref<?x?x?xf64>
                %C4_val = affine.load %C4[%ss + %s, %pp + %p] : memref<?x?xf64>
                %product = arith.mulf %A_val, %C4_val : f64
                %current_sum = affine.load %sum[%pp + %p] : memref<?xf64>
                %new_sum = arith.addf %current_sum, %product : f64
                affine.store %new_sum, %sum[%pp + %p] : memref<?xf64>
              }
            }
          }
        }
        // Write back A[r][q][p] = sum[p]
        affine.for %p = 0 to %NP {
          %final_value = affine.load %sum[%p] : memref<?xf64>
          affine.store %final_value, %A[%r, %q, %p] : memref<?x?x?xf64>
        }
      }
    } {dmd.extract}
    return
  }
}
