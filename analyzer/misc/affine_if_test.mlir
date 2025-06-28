#set = affine_set<(i)[] : (i - 5 >= 0)>
module {
  func.func @simple_affine_nested_loops(%arg0: memref<10xi32>) {
    affine.for %j = 0 to 10 {
      affine.for %i = 0 to 10 {
        affine.if #set(%i)[] {
          %c1 = arith.constant 1 : i32
          affine.store %c1, %arg0[%i] : memref<10xi32>
        }
      }
    }
    return
  }
}
