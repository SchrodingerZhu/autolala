#lb = affine_map<(d0) -> (d0)>
#ub32 = affine_map<(d0)[s0] -> (d0 + 32, s0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    // Small N (192-384): B rows are short enough that a single full j sweep
    // stays cache-resident. Interchange to i-k-j (j unit-stride; A[i,k]
    // register-invariant across j) with a light i-tile to keep active C rows
    // hot. No j/k tiling -> minimal loop overhead at small sizes.
    affine.for %ii = 0 to %N step 32 {
      affine.for %i = #lb(%ii) to min #ub32(%ii)[%N] {
        affine.for %k = 0 to %N {
          %a = affine.load %A[%i, %k] : memref<?x?xf64>
          affine.for %j = 0 to %N {
            %b = affine.load %B[%k, %j] : memref<?x?xf64>
            %c = affine.load %C[%i, %j] : memref<?x?xf64>
            %p = arith.mulf %a, %b : f64
            %s = arith.addf %c, %p : f64
            affine.store %s, %C[%i, %j] : memref<?x?xf64>
          }
        }
      }
    }
    return
  }
}
