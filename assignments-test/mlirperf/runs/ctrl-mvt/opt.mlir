#lb = affine_map<(d0) -> (d0)>
#ub = affine_map<(d0)[s0] -> (d0 + 512, s0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %x1: memref<?xf64>, %x2: memref<?xf64>, %y1: memref<?xf64>, %y2: memref<?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    // Pass 1: x1[i] += A[i,j] * y1[j]. A row-major. Tile j so a y1-block stays hot
    // across the full i sweep; x1[i] accumulates partial sums per j-tile.
    affine.for %jj = 0 to %N step 512 {
      affine.for %i = 0 to %N {
        affine.for %j = #lb(%jj) to min #ub(%jj)[%N] {
          %x = affine.load %x1[%i] : memref<?xf64>
          %a = affine.load %A[%i,%j] : memref<?x?xf64>
          %y = affine.load %y1[%j] : memref<?xf64>
          %p = arith.mulf %a, %y : f64
          %s = arith.addf %x, %p : f64
          affine.store %s, %x1[%i] : memref<?xf64>
        }
      }
    }
    // Pass 2: x2[i] += A[j,i] * y2[j]. Interchange so i is inner (A[j,i] row-major).
    // Tile i so an x2-block stays resident across the full j sweep (its reuse axis).
    affine.for %ii = 0 to %N step 512 {
      affine.for %j = 0 to %N {
        affine.for %i = #lb(%ii) to min #ub(%ii)[%N] {
          %x = affine.load %x2[%i] : memref<?xf64>
          %a = affine.load %A[%j,%i] : memref<?x?xf64>
          %y = affine.load %y2[%j] : memref<?xf64>
          %p = arith.mulf %a, %y : f64
          %s = arith.addf %x, %p : f64
          affine.store %s, %x2[%i] : memref<?xf64>
        }
      }
    }
    return
  }
}
