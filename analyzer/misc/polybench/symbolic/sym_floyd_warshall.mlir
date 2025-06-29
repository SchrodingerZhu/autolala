module {
  func.func @kernel_floyd_warshall(%path: memref<?x?xf64>, %N : index) {
    
    // Triple nested loop: k, i, j all from 0 to %N
    affine.for %k = 0 to %N {
      affine.for %i = 0 to %N {
        affine.for %j = 0 to %N {
          // Load path[i][j], path[i][k], path[k][j]
          %path_ij = affine.load %path[%i, %j] : memref<?x?xf64>
          %path_ik = affine.load %path[%i, %k] : memref<?x?xf64>
          %path_kj = affine.load %path[%k, %j] : memref<?x?xf64>
          
          // Compute path[i][k] + path[k][j]
          %sum = arith.addf %path_ik, %path_kj : f64
          
          // Compare: path[i][j] < path[i][k] + path[k][j]
          %cond = arith.cmpf olt, %path_ij, %sum : f64
          
          // Select minimum: path[i][j] < sum ? path[i][j] : sum
          %min_val = arith.select %cond, %path_ij, %sum : f64
          
          // Store result back to path[i][j]
          affine.store %min_val, %path[%i, %j] : memref<?x?xf64>
        }
      }
    }
    
    return
  }
}
