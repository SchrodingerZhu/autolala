module {
  // GEMM C = 0.9*C + A*B, tiled i-k-j + unroll-and-jam x4 on i.
  // Tiles: iT=64 kT=256 jT=96. UJ covers i<floor4(N); bottom rows separate.
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64

    affine.for %i = 0 to %N {
      affine.for %j = 0 to %N {
        %c = affine.load %C[%i,%j] : memref<?x?xf64>
        %cs = arith.mulf %c, %b9 : f64
        affine.store %cs, %C[%i,%j] : memref<?x?xf64>
      }
    }

    affine.for %ii = 0 to affine_map<()[s0] -> (s0 floordiv 4 * 4)>()[%N] step 64 {
      affine.for %kk = 0 to %N step 256 {
        affine.for %jj = 0 to %N step 96 {
          affine.for %i = affine_map<(d0) -> (d0)>(%ii)
              to min affine_map<(d0)[s0] -> (d0 + 64, s0 floordiv 4 * 4)>(%ii)[%N] step 4 {
            affine.for %k = affine_map<(d0) -> (d0)>(%kk) to min affine_map<(d0)[s0] -> (d0 + 256, s0)>(%kk)[%N] {
              %a0 = affine.load %A[%i,%k] : memref<?x?xf64>
              %a1 = affine.load %A[%i+1,%k] : memref<?x?xf64>
              %a2 = affine.load %A[%i+2,%k] : memref<?x?xf64>
              %a3 = affine.load %A[%i+3,%k] : memref<?x?xf64>
              affine.for %j = affine_map<(d0) -> (d0)>(%jj) to min affine_map<(d0)[s0] -> (d0 + 96, s0)>(%jj)[%N] {
                %b = affine.load %B[%k,%j] : memref<?x?xf64>
                %c0 = affine.load %C[%i,%j] : memref<?x?xf64>
                %c1 = affine.load %C[%i+1,%j] : memref<?x?xf64>
                %c2 = affine.load %C[%i+2,%j] : memref<?x?xf64>
                %c3 = affine.load %C[%i+3,%j] : memref<?x?xf64>
                %p0 = arith.mulf %a0, %b : f64
                %p1 = arith.mulf %a1, %b : f64
                %p2 = arith.mulf %a2, %b : f64
                %p3 = arith.mulf %a3, %b : f64
                %s0 = arith.addf %c0, %p0 : f64
                %s1 = arith.addf %c1, %p1 : f64
                %s2 = arith.addf %c2, %p2 : f64
                %s3 = arith.addf %c3, %p3 : f64
                affine.store %s0, %C[%i,%j] : memref<?x?xf64>
                affine.store %s1, %C[%i+1,%j] : memref<?x?xf64>
                affine.store %s2, %C[%i+2,%j] : memref<?x?xf64>
                affine.store %s3, %C[%i+3,%j] : memref<?x?xf64>
              }
            }
          }
        }
      }
    }

    affine.for %i = affine_map<()[s0] -> (s0 floordiv 4 * 4)>()[%N] to %N {
      affine.for %kk = 0 to %N step 256 {
        affine.for %k = affine_map<(d0) -> (d0)>(%kk) to min affine_map<(d0)[s0] -> (d0 + 256, s0)>(%kk)[%N] {
          %a0 = affine.load %A[%i,%k] : memref<?x?xf64>
          affine.for %j = 0 to %N {
            %b = affine.load %B[%k,%j] : memref<?x?xf64>
            %c0 = affine.load %C[%i,%j] : memref<?x?xf64>
            %p0 = arith.mulf %a0, %b : f64
            %s0 = arith.addf %c0, %p0 : f64
            affine.store %s0, %C[%i,%j] : memref<?x?xf64>
          }
        }
      }
    }
    return
  }
}
