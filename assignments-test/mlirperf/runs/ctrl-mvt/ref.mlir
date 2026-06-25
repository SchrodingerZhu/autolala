module {
  func.func @kernel(%A: memref<?x?xf64>, %x1: memref<?xf64>, %x2: memref<?xf64>, %y1: memref<?xf64>, %y2: memref<?xf64>, %N: index) attributes {llvm.emit_c_interface} {
    affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %x = affine.load %x1[%i] : memref<?xf64>
      %a = affine.load %A[%i,%j] : memref<?x?xf64>
      %y = affine.load %y1[%j] : memref<?xf64>
      %p = arith.mulf %a, %y : f64
      %s = arith.addf %x, %p : f64
      affine.store %s, %x1[%i] : memref<?xf64>
    }}
    affine.for %i = 0 to %N { affine.for %j = 0 to %N {
      %x = affine.load %x2[%i] : memref<?xf64>
      %a = affine.load %A[%j,%i] : memref<?x?xf64>
      %y = affine.load %y2[%j] : memref<?xf64>
      %p = arith.mulf %a, %y : f64
      %s = arith.addf %x, %p : f64
      affine.store %s, %x2[%i] : memref<?xf64>
    }}
    return
  }
}
