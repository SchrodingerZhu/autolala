#tri = affine_map<(d0) -> (d0 + 1)>
#p1 = affine_map<(d0) -> (d0 + 1)>
#p2 = affine_map<(d0) -> (d0 + 2)>
#p3 = affine_map<(d0) -> (d0 + 3)>
// guard: ii + r < N   i.e.  N - ii - r - 1 >= 0
#row1 = affine_set<(d0)[s0] : (s0 - d0 - 2 >= 0)>
#row2 = affine_set<(d0)[s0] : (s0 - d0 - 3 >= 0)>
#row3 = affine_set<(d0)[s0] : (s0 - d0 - 4 >= 0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64

    // pre-scale lower triangle of C by 0.9
    affine.for %i = 0 to %N {
      affine.for %j = 0 to #tri(%i) {
        %c = affine.load %C[%i,%j] : memref<?x?xf64>
        %cs = arith.mulf %c, %b9 : f64
        affine.store %cs, %C[%i,%j] : memref<?x?xf64>
      }
    }

    // Main compute: C[i,j] += sum_k A[i,k]*A[j,k]  for j<=i.  Order i-k-j.
    // Unroll-and-jam the i loop by 4 (block [ii, ii+4)). For the inner j the
    // rectangular range [0, ii] is valid for ALL rows in the block (since the
    // smallest row index is ii, and j<=ii<=i). Each diagonal/triangular entry
    // (rows where j>ii) is handled by per-row guarded stores below.
    affine.for %ii = 0 to %N step 4 {
      %i1 = affine.apply #p1(%ii)
      %i2 = affine.apply #p2(%ii)
      %i3 = affine.apply #p3(%ii)
      affine.for %k = 0 to %N {
        %a0 = affine.load %A[%ii,%k] : memref<?x?xf64>
        %a1 = affine.if #row1(%ii)[%N] -> f64 {
          %v = affine.load %A[%i1,%k] : memref<?x?xf64>
          affine.yield %v : f64
        } else {
          %z = arith.constant 0.0 : f64
          affine.yield %z : f64
        }
        %a2 = affine.if #row2(%ii)[%N] -> f64 {
          %v = affine.load %A[%i2,%k] : memref<?x?xf64>
          affine.yield %v : f64
        } else {
          %z = arith.constant 0.0 : f64
          affine.yield %z : f64
        }
        %a3 = affine.if #row3(%ii)[%N] -> f64 {
          %v = affine.load %A[%i3,%k] : memref<?x?xf64>
          affine.yield %v : f64
        } else {
          %z = arith.constant 0.0 : f64
          affine.yield %z : f64
        }
        // rectangular part: j in [0, ii], valid for every row.
        affine.for %j = 0 to #tri(%ii) {
          %ajk = affine.load %A[%j,%k] : memref<?x?xf64>
          // row ii
          %c0 = affine.load %C[%ii,%j] : memref<?x?xf64>
          %p0 = arith.mulf %a0, %ajk : f64
          %s0 = arith.addf %c0, %p0 : f64
          affine.store %s0, %C[%ii,%j] : memref<?x?xf64>
          // row ii+1
          affine.if #row1(%ii)[%N] {
            %c = affine.load %C[%i1,%j] : memref<?x?xf64>
            %p = arith.mulf %a1, %ajk : f64
            %s = arith.addf %c, %p : f64
            affine.store %s, %C[%i1,%j] : memref<?x?xf64>
          }
          // row ii+2
          affine.if #row2(%ii)[%N] {
            %c = affine.load %C[%i2,%j] : memref<?x?xf64>
            %p = arith.mulf %a2, %ajk : f64
            %s = arith.addf %c, %p : f64
            affine.store %s, %C[%i2,%j] : memref<?x?xf64>
          }
          // row ii+3
          affine.if #row3(%ii)[%N] {
            %c = affine.load %C[%i3,%j] : memref<?x?xf64>
            %p = arith.mulf %a3, %ajk : f64
            %s = arith.addf %c, %p : f64
            affine.store %s, %C[%i3,%j] : memref<?x?xf64>
          }
        }
        // triangular cleanup: entries with j in (ii, row].
        // row ii+1 contributes j = ii+1
        affine.if #row1(%ii)[%N] {
          %ajk = affine.load %A[%i1,%k] : memref<?x?xf64>
          %c = affine.load %C[%i1,%i1] : memref<?x?xf64>
          %p = arith.mulf %a1, %ajk : f64
          %s = arith.addf %c, %p : f64
          affine.store %s, %C[%i1,%i1] : memref<?x?xf64>
        }
        // row ii+2 contributes j = ii+1, ii+2
        affine.if #row2(%ii)[%N] {
          %aj1 = affine.load %A[%i1,%k] : memref<?x?xf64>
          %c1v = affine.load %C[%i2,%i1] : memref<?x?xf64>
          %p1 = arith.mulf %a2, %aj1 : f64
          %s1 = arith.addf %c1v, %p1 : f64
          affine.store %s1, %C[%i2,%i1] : memref<?x?xf64>
          %aj2 = affine.load %A[%i2,%k] : memref<?x?xf64>
          %c2v = affine.load %C[%i2,%i2] : memref<?x?xf64>
          %p2 = arith.mulf %a2, %aj2 : f64
          %s2 = arith.addf %c2v, %p2 : f64
          affine.store %s2, %C[%i2,%i2] : memref<?x?xf64>
        }
        // row ii+3 contributes j = ii+1, ii+2, ii+3
        affine.if #row3(%ii)[%N] {
          %aj1 = affine.load %A[%i1,%k] : memref<?x?xf64>
          %c1v = affine.load %C[%i3,%i1] : memref<?x?xf64>
          %p1 = arith.mulf %a3, %aj1 : f64
          %s1 = arith.addf %c1v, %p1 : f64
          affine.store %s1, %C[%i3,%i1] : memref<?x?xf64>
          %aj2 = affine.load %A[%i2,%k] : memref<?x?xf64>
          %c2v = affine.load %C[%i3,%i2] : memref<?x?xf64>
          %p2 = arith.mulf %a3, %aj2 : f64
          %s2 = arith.addf %c2v, %p2 : f64
          affine.store %s2, %C[%i3,%i2] : memref<?x?xf64>
          %aj3 = affine.load %A[%i3,%k] : memref<?x?xf64>
          %c3v = affine.load %C[%i3,%i3] : memref<?x?xf64>
          %p3 = arith.mulf %a3, %aj3 : f64
          %s3 = arith.addf %c3v, %p3 : f64
          affine.store %s3, %C[%i3,%i3] : memref<?x?xf64>
        }
      }
    }
    return
  }
}
