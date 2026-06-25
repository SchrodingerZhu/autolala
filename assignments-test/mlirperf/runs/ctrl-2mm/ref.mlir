module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %D: memref<?x?xf64>, %T: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64
    affine.for %i = 0 to %N { affine.for %j = 0 to %N { affine.for %k = 0 to %N {
      %a = affine.load %A[%i,%k] : memref<?x?xf64>
      %b = affine.load %B[%k,%j] : memref<?x?xf64>
      %t = affine.load %T[%i,%j] : memref<?x?xf64>
      %p = arith.mulf %a, %b : f64
      %s = arith.addf %t, %p : f64
      affine.store %s, %T[%i,%j] : memref<?x?xf64>
    }}}
    affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %d = affine.load %D[%i,%j] : memref<?x?xf64>
      %ds = arith.mulf %d, %b9 : f64
      affine.store %ds, %D[%i,%j] : memref<?x?xf64>
    }}
    affine.for %i = 0 to %N { affine.for %j = 0 to %N { affine.for %k = 0 to %N {
      %t = affine.load %T[%i,%k] : memref<?x?xf64>
      %c = affine.load %C[%k,%j] : memref<?x?xf64>
      %d = affine.load %D[%i,%j] : memref<?x?xf64>
      %p = arith.mulf %t, %c : f64
      %s = arith.addf %d, %p : f64
      affine.store %s, %D[%i,%j] : memref<?x?xf64>
    }}}
    return
  }
}
