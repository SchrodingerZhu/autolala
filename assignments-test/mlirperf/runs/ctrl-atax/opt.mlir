#ub8 = affine_map<(d0)[s0] -> (d0 + 8, s0)>
#main = affine_map<(d0) -> ((d0 floordiv 8) * 8)>
module {
  // atax, fused two-phase with i-blocking (TI=8) + unroll-and-jam on the j sweep.
  //  pass1: T[i]=sum_j A[i,j]*x[j]  (A read once, x reused, j unit-stride -> vectorizes)
  //  pass2: y[j]+=sum_{i in block} A[i,j]*T[i]  jammed over 8 rows so each y[j] is
  //         loaded/stored once per block (y traffic /8) and j stays unit-stride (vectorizes);
  //         the 8 A[i,j] streams are each unit-stride in j.
  func.func @kernel(%A: memref<?x?xf64>, %x: memref<?xf64>, %y: memref<?xf64>, %T: memref<?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0.0 : f64
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index

    // ---- full blocks of 8 rows ----
    affine.for %ii = 0 to #main(%N)[] step 8 {
      %i1 = arith.addi %ii, %c1 : index
      %i2 = arith.addi %ii, %c2 : index
      %i3 = arith.addi %ii, %c3 : index
      %i4 = arith.addi %ii, %c4 : index
      %i5 = arith.addi %ii, %c5 : index
      %i6 = arith.addi %ii, %c6 : index
      %i7 = arith.addi %ii, %c7 : index
      // pass 1 (jammed over 8 rows): x[j] loaded once/ row-block, 8 independent
      // accumulators give scalar ILP; T[i]=sum_j A[i,j]*x[j].
      %r:8 = affine.for %j = 0 to %N
          iter_args(%b0=%c0,%b1=%c0,%b2=%c0,%b3=%c0,%b4=%c0,%b5=%c0,%b6=%c0,%b7=%c0)
          -> (f64,f64,f64,f64,f64,f64,f64,f64) {
        %xv = affine.load %x[%j] : memref<?xf64>
        %m0 = affine.load %A[%ii, %j] : memref<?x?xf64>
        %m1 = memref.load %A[%i1, %j] : memref<?x?xf64>
        %m2 = memref.load %A[%i2, %j] : memref<?x?xf64>
        %m3 = memref.load %A[%i3, %j] : memref<?x?xf64>
        %m4 = memref.load %A[%i4, %j] : memref<?x?xf64>
        %m5 = memref.load %A[%i5, %j] : memref<?x?xf64>
        %m6 = memref.load %A[%i6, %j] : memref<?x?xf64>
        %m7 = memref.load %A[%i7, %j] : memref<?x?xf64>
        %n0 = arith.mulf %m0, %xv : f64
        %n1 = arith.mulf %m1, %xv : f64
        %n2 = arith.mulf %m2, %xv : f64
        %n3 = arith.mulf %m3, %xv : f64
        %n4 = arith.mulf %m4, %xv : f64
        %n5 = arith.mulf %m5, %xv : f64
        %n6 = arith.mulf %m6, %xv : f64
        %n7 = arith.mulf %m7, %xv : f64
        %e0 = arith.addf %b0, %n0 : f64
        %e1 = arith.addf %b1, %n1 : f64
        %e2 = arith.addf %b2, %n2 : f64
        %e3 = arith.addf %b3, %n3 : f64
        %e4 = arith.addf %b4, %n4 : f64
        %e5 = arith.addf %b5, %n5 : f64
        %e6 = arith.addf %b6, %n6 : f64
        %e7 = arith.addf %b7, %n7 : f64
        affine.yield %e0,%e1,%e2,%e3,%e4,%e5,%e6,%e7 : f64,f64,f64,f64,f64,f64,f64,f64
      }
      affine.store %r#0, %T[%ii] : memref<?xf64>
      memref.store %r#1, %T[%i1] : memref<?xf64>
      memref.store %r#2, %T[%i2] : memref<?xf64>
      memref.store %r#3, %T[%i3] : memref<?xf64>
      memref.store %r#4, %T[%i4] : memref<?xf64>
      memref.store %r#5, %T[%i5] : memref<?xf64>
      memref.store %r#6, %T[%i6] : memref<?xf64>
      memref.store %r#7, %T[%i7] : memref<?xf64>
      // pass 2 (jammed)
      affine.for %j = 0 to %N {
        %yv = affine.load %y[%j] : memref<?xf64>
        %a0 = affine.load %A[%ii, %j] : memref<?x?xf64>
        %a1 = memref.load %A[%i1, %j] : memref<?x?xf64>
        %a2 = memref.load %A[%i2, %j] : memref<?x?xf64>
        %a3 = memref.load %A[%i3, %j] : memref<?x?xf64>
        %a4 = memref.load %A[%i4, %j] : memref<?x?xf64>
        %a5 = memref.load %A[%i5, %j] : memref<?x?xf64>
        %a6 = memref.load %A[%i6, %j] : memref<?x?xf64>
        %a7 = memref.load %A[%i7, %j] : memref<?x?xf64>
        %p0 = arith.mulf %a0, %r#0 : f64
        %p1 = arith.mulf %a1, %r#1 : f64
        %p2 = arith.mulf %a2, %r#2 : f64
        %p3 = arith.mulf %a3, %r#3 : f64
        %p4 = arith.mulf %a4, %r#4 : f64
        %p5 = arith.mulf %a5, %r#5 : f64
        %p6 = arith.mulf %a6, %r#6 : f64
        %p7 = arith.mulf %a7, %r#7 : f64
        // balanced reduction tree (benign reassociation) for scalar ILP
        %q0 = arith.addf %p0, %p1 : f64
        %q1 = arith.addf %p2, %p3 : f64
        %q2 = arith.addf %p4, %p5 : f64
        %q3 = arith.addf %p6, %p7 : f64
        %w0 = arith.addf %q0, %q1 : f64
        %w1 = arith.addf %q2, %q3 : f64
        %w2 = arith.addf %w0, %w1 : f64
        %s7 = arith.addf %yv, %w2 : f64
        affine.store %s7, %y[%j] : memref<?xf64>
      }
    }

    // ---- remainder rows (0..7 of them), unfused, simple ----
    affine.for %i = #main(%N)[] to %N {
      %ti = affine.for %j = 0 to %N iter_args(%acc = %c0) -> (f64) {
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
    return
  }
}
