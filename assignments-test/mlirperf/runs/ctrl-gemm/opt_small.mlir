module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64
    // Fuse the 0.9 scaling into the k=0 step of the GEMM via interchanged i,k,j.
    // For small N the whole B fits in cache; plain interchange (no tiling) avoids
    // loop-overhead and gives unit-stride inner j with A[i,k] reused across j.
    affine.for %i = 0 to %N {
      affine.for %j = 0 to %N {
        %c = affine.load %C[%i,%j] : memref<?x?xf64>
        %cs = arith.mulf %c, %b9 : f64
        affine.store %cs, %C[%i,%j] : memref<?x?xf64>
      }
    }
    affine.for %i = 0 to %N {
      affine.for %k = 0 to %N {
        %a = affine.load %A[%i,%k] : memref<?x?xf64>
        affine.for %j = 0 to %N {
          %b = affine.load %B[%k,%j] : memref<?x?xf64>
          %c = affine.load %C[%i,%j] : memref<?x?xf64>
          %p = arith.mulf %a, %b : f64
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%i,%j] : memref<?x?xf64>
        }
      }
    }
    return
  }
}
