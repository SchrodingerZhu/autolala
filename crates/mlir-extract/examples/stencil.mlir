// 2D Jacobi-style stencil on static 100x100 arrays.
module {
  func.func @stencil(%A: memref<100x100xf32>, %B: memref<100x100xf32>) {
    affine.for %i = 1 to 99 {
      affine.for %j = 1 to 99 {
        %t = affine.load %A[%i - 1, %j] : memref<100x100xf32>
        %b = affine.load %A[%i + 1, %j] : memref<100x100xf32>
        %l = affine.load %A[%i, %j - 1] : memref<100x100xf32>
        %r = affine.load %A[%i, %j + 1] : memref<100x100xf32>
        %c = affine.load %A[%i, %j] : memref<100x100xf32>
        %s1 = arith.addf %t, %b : f32
        %s2 = arith.addf %l, %r : f32
        %s3 = arith.addf %s1, %s2 : f32
        %s4 = arith.addf %s3, %c : f32
        affine.store %s4, %B[%i, %j] : memref<100x100xf32>
      }
    } { dmd.extract }
    return
  }
}
