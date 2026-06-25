#tail_lb = affine_map<(d0) -> (d0)>
#tail_ub = affine_map<(d0)[s0] -> (s0, d0 + 8)>
module {
  func.func @kernel(%A: memref<?x?xf64>, %x: memref<?xf64>, %y: memref<?xf64>, %T: memref<?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %cu = arith.constant 8 : index
    %zero = arith.constant 0.0 : f64
    affine.for %ii = 0 to %N step 8 {
      %rem = affine.apply affine_map<(d0)[s0] -> (s0 - d0)>(%ii)[%N]
      %rem_ge = arith.cmpi sge, %rem, %cu : index
      scf.if %rem_ge {
        %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%ii)
        %i2 = affine.apply affine_map<(d0) -> (d0 + 2)>(%ii)
        %i3 = affine.apply affine_map<(d0) -> (d0 + 3)>(%ii)
        %i4 = affine.apply affine_map<(d0) -> (d0 + 4)>(%ii)
        %i5 = affine.apply affine_map<(d0) -> (d0 + 5)>(%ii)
        %i6 = affine.apply affine_map<(d0) -> (d0 + 6)>(%ii)
        %i7 = affine.apply affine_map<(d0) -> (d0 + 7)>(%ii)
        %t0 = affine.for %j = 0 to %N iter_args(%acc = %zero) -> (f64) {
          %a = affine.load %A[%ii, %j] : memref<?x?xf64>
          %xv = affine.load %x[%j] : memref<?xf64>
          %p = arith.mulf %a, %xv : f64
          %s = arith.addf %acc, %p : f64
          affine.yield %s : f64
        }
        affine.store %t0, %T[%ii] : memref<?xf64>
        %t1 = affine.for %j = 0 to %N iter_args(%acc = %zero) -> (f64) {
          %a = affine.load %A[%i1, %j] : memref<?x?xf64>
          %xv = affine.load %x[%j] : memref<?xf64>
          %p = arith.mulf %a, %xv : f64
          %s = arith.addf %acc, %p : f64
          affine.yield %s : f64
        }
        affine.store %t1, %T[%i1] : memref<?xf64>
        %t2 = affine.for %j = 0 to %N iter_args(%acc = %zero) -> (f64) {
          %a = affine.load %A[%i2, %j] : memref<?x?xf64>
          %xv = affine.load %x[%j] : memref<?xf64>
          %p = arith.mulf %a, %xv : f64
          %s = arith.addf %acc, %p : f64
          affine.yield %s : f64
        }
        affine.store %t2, %T[%i2] : memref<?xf64>
        %t3 = affine.for %j = 0 to %N iter_args(%acc = %zero) -> (f64) {
          %a = affine.load %A[%i3, %j] : memref<?x?xf64>
          %xv = affine.load %x[%j] : memref<?xf64>
          %p = arith.mulf %a, %xv : f64
          %s = arith.addf %acc, %p : f64
          affine.yield %s : f64
        }
        affine.store %t3, %T[%i3] : memref<?xf64>
        %t4 = affine.for %j = 0 to %N iter_args(%acc = %zero) -> (f64) {
          %a = affine.load %A[%i4, %j] : memref<?x?xf64>
          %xv = affine.load %x[%j] : memref<?xf64>
          %p = arith.mulf %a, %xv : f64
          %s = arith.addf %acc, %p : f64
          affine.yield %s : f64
        }
        affine.store %t4, %T[%i4] : memref<?xf64>
        %t5 = affine.for %j = 0 to %N iter_args(%acc = %zero) -> (f64) {
          %a = affine.load %A[%i5, %j] : memref<?x?xf64>
          %xv = affine.load %x[%j] : memref<?xf64>
          %p = arith.mulf %a, %xv : f64
          %s = arith.addf %acc, %p : f64
          affine.yield %s : f64
        }
        affine.store %t5, %T[%i5] : memref<?xf64>
        %t6 = affine.for %j = 0 to %N iter_args(%acc = %zero) -> (f64) {
          %a = affine.load %A[%i6, %j] : memref<?x?xf64>
          %xv = affine.load %x[%j] : memref<?xf64>
          %p = arith.mulf %a, %xv : f64
          %s = arith.addf %acc, %p : f64
          affine.yield %s : f64
        }
        affine.store %t6, %T[%i6] : memref<?xf64>
        %t7 = affine.for %j = 0 to %N iter_args(%acc = %zero) -> (f64) {
          %a = affine.load %A[%i7, %j] : memref<?x?xf64>
          %xv = affine.load %x[%j] : memref<?xf64>
          %p = arith.mulf %a, %xv : f64
          %s = arith.addf %acc, %p : f64
          affine.yield %s : f64
        }
        affine.store %t7, %T[%i7] : memref<?xf64>
        affine.for %j = 0 to %N {
          %yv = affine.load %y[%j] : memref<?xf64>
          %a0 = affine.load %A[%ii, %j] : memref<?x?xf64>
          %a1 = affine.load %A[%i1, %j] : memref<?x?xf64>
          %a2 = affine.load %A[%i2, %j] : memref<?x?xf64>
          %a3 = affine.load %A[%i3, %j] : memref<?x?xf64>
          %a4 = affine.load %A[%i4, %j] : memref<?x?xf64>
          %a5 = affine.load %A[%i5, %j] : memref<?x?xf64>
          %a6 = affine.load %A[%i6, %j] : memref<?x?xf64>
          %a7 = affine.load %A[%i7, %j] : memref<?x?xf64>
          %p0 = arith.mulf %a0, %t0 : f64
          %p1 = arith.mulf %a1, %t1 : f64
          %p2 = arith.mulf %a2, %t2 : f64
          %p3 = arith.mulf %a3, %t3 : f64
          %p4 = arith.mulf %a4, %t4 : f64
          %p5 = arith.mulf %a5, %t5 : f64
          %p6 = arith.mulf %a6, %t6 : f64
          %p7 = arith.mulf %a7, %t7 : f64
          %q01 = arith.addf %p0, %p1 : f64
          %q23 = arith.addf %p2, %p3 : f64
          %q45 = arith.addf %p4, %p5 : f64
          %q67 = arith.addf %p6, %p7 : f64
          %q0123 = arith.addf %q01, %q23 : f64
          %q4567 = arith.addf %q45, %q67 : f64
          %qsum = arith.addf %q0123, %q4567 : f64
          %s7 = arith.addf %yv, %qsum : f64
          affine.store %s7, %y[%j] : memref<?xf64>
        }
      } else {
        affine.for %i = #tail_lb(%ii) to min #tail_ub(%ii)[%N] {
          %ti = affine.for %j = 0 to %N iter_args(%acc = %zero) -> (f64) {
            %a = affine.load %A[%i, %j] : memref<?x?xf64>
            %xv = affine.load %x[%j] : memref<?xf64>
            %p = arith.mulf %a, %xv : f64
            %s = arith.addf %acc, %p : f64
            affine.yield %s : f64
          }
          affine.store %ti, %T[%i] : memref<?xf64>
          affine.for %j = 0 to %N {
            %yv = affine.load %y[%j] : memref<?xf64>
            %a = affine.load %A[%i, %j] : memref<?x?xf64>
            %p = arith.mulf %a, %ti : f64
            %s = arith.addf %yv, %p : f64
            affine.store %s, %y[%j] : memref<?xf64>
          }
        }
      }
    }
    return
  }
}

