module {
  func.func @kernel(%A: memref<?x?xf64>, %x: memref<?xf64>, %y: memref<?xf64>, %T: memref<?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %tv = affine.load %T[%i] : memref<?xf64>
      %a = affine.load %A[%i,%j] : memref<?x?xf64>
      %xv = affine.load %x[%j] : memref<?xf64>
      %p = arith.mulf %a, %xv : f64
      %s = arith.addf %tv, %p : f64
      affine.store %s, %T[%i] : memref<?xf64>
    }}
    affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %yv = affine.load %y[%j] : memref<?xf64>
      %a = affine.load %A[%i,%j] : memref<?x?xf64>
      %tv = affine.load %T[%i] : memref<?xf64>
      %p = arith.mulf %a, %tv : f64
      %s = arith.addf %yv, %p : f64
      affine.store %s, %y[%j] : memref<?xf64>
    }}
    return
  }
}
