module {
  func.func @kernel(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    %b9 = arith.constant 0.9 : f64
    affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %c = affine.load %C[%i,%j] : memref<?x?xf64>
      %cs = arith.mulf %c, %b9 : f64
      affine.store %cs, %C[%i,%j] : memref<?x?xf64>
    }}
    affine.for %i = 0 to %N { affine.for %j = 0 to %N { affine.for %k = 0 to %N {
      %a = affine.load %A[%i,%k] : memref<?x?xf64>
      %b = affine.load %B[%k,%j] : memref<?x?xf64>
      %c = affine.load %C[%i,%j] : memref<?x?xf64>
      %p = arith.mulf %a, %b : f64
      %s = arith.addf %c, %p : f64
      affine.store %s, %C[%i,%j] : memref<?x?xf64>
    }}}
    return
  }
}
