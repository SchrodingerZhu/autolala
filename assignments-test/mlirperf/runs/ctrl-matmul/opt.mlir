#lb   = affine_map<(d0) -> (d0)>
#ub_k = affine_map<(d0)[s0] -> (d0 + 128, s0)>
#ub_j = affine_map<(d0)[s0] -> (d0 + 512, s0)>
#i4   = affine_map<(d0)[s0] -> (d0 + 4, s0)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    // Cache-tiled matmul, i-k-j interchange, with i unrolled-and-jammed x4.
    // Tile k by 128, j by 512.  For each k-tile we keep a B-panel resident and
    // reuse it across all i-rows.  Unrolling i by 4 lets each B[k,j] load feed
    // 4 FMAs (4 C rows held in registers) -> cuts B traffic 4x in the hot loop.
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    affine.for %kc = 0 to %N step 128 {
      affine.for %jc = 0 to %N step 512 {
        // i blocked by 4 (unroll-and-jam); compute i upper bound that is a
        // multiple of 4, handle the tail separately.
        %nb = arith.subi %N, %c0 : index
        affine.for %ii = 0 to %N step 4 {
          %i0 = arith.addi %ii, %c0 : index
          %i1 = arith.addi %ii, %c1 : index
          %rem = arith.subi %N, %ii : index
          %full = arith.cmpi sge, %rem, %c4 : index
          scf.if %full {
            %i2 = arith.addi %ii, %c1 : index
            affine.for %k = #lb(%kc) to min #ub_k(%kc)[%N] {
              %a0 = affine.load %A[%ii, %k] : memref<?x?xf64>
              %a1 = affine.load %A[%ii + 1, %k] : memref<?x?xf64>
              %a2 = affine.load %A[%ii + 2, %k] : memref<?x?xf64>
              %a3 = affine.load %A[%ii + 3, %k] : memref<?x?xf64>
              affine.for %j = #lb(%jc) to min #ub_j(%jc)[%N] {
                %b = affine.load %B[%k, %j] : memref<?x?xf64>
                %c0v = affine.load %C[%ii, %j] : memref<?x?xf64>
                %c1v = affine.load %C[%ii + 1, %j] : memref<?x?xf64>
                %c2v = affine.load %C[%ii + 2, %j] : memref<?x?xf64>
                %c3v = affine.load %C[%ii + 3, %j] : memref<?x?xf64>
                %p0 = arith.mulf %a0, %b : f64
                %p1 = arith.mulf %a1, %b : f64
                %p2 = arith.mulf %a2, %b : f64
                %p3 = arith.mulf %a3, %b : f64
                %s0 = arith.addf %c0v, %p0 : f64
                %s1 = arith.addf %c1v, %p1 : f64
                %s2 = arith.addf %c2v, %p2 : f64
                %s3 = arith.addf %c3v, %p3 : f64
                affine.store %s0, %C[%ii, %j] : memref<?x?xf64>
                affine.store %s1, %C[%ii + 1, %j] : memref<?x?xf64>
                affine.store %s2, %C[%ii + 2, %j] : memref<?x?xf64>
                affine.store %s3, %C[%ii + 3, %j] : memref<?x?xf64>
              }
            }
          } else {
            // tail: 1..3 remaining rows, scalar over the same tiles
            affine.for %it = #lb(%ii) to %N {
              affine.for %k = #lb(%kc) to min #ub_k(%kc)[%N] {
                %a = affine.load %A[%it, %k] : memref<?x?xf64>
                affine.for %j = #lb(%jc) to min #ub_j(%jc)[%N] {
                  %b = affine.load %B[%k, %j] : memref<?x?xf64>
                  %c = affine.load %C[%it, %j] : memref<?x?xf64>
                  %p = arith.mulf %a, %b : f64
                  %s = arith.addf %c, %p : f64
                  affine.store %s, %C[%it, %j] : memref<?x?xf64>
                }
              }
            }
          }
        }
      }
    }
    return
  }
}
