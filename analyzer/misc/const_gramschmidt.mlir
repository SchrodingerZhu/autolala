module {
  // A  : memref<?x?xf32>  (M × N，M = 128, N = 128)
  // R  : memref<?x?xf32>  (N × N)
  // Q  : memref<?x?xf32>  (M × N)
  func.func @gramschmidt(%A: memref<?x?xf32>,
                         %R: memref<?x?xf32>,
                         %Q: memref<?x?xf32>) {
    // 常量 0.0f
    %cst0 = arith.constant 0.0 : f32

    // k 循环：列索引
    affine.for %k = 0 to 128 {
      // ─────── 1. 计算列范数 nrm = Σ_i A[i,k]² ───────
      %nrm = affine.for %i = 0 to 128 iter_args(%acc = %cst0) -> (f32) {
        %Aik = affine.load %A[%i, %k] : memref<?x?xf32>
        %sq  = arith.mulf %Aik, %Aik : f32
        %sum = arith.addf %acc, %sq  : f32
        affine.yield %sum : f32
      }
      // R[k,k] = sqrt(nrm)
      %Rkk = math.sqrt %nrm : f32
      affine.store %Rkk, %R[%k, %k] : memref<?x?xf32>

      // ─────── 2. 归一化得到 Q[:,k] ───────
      affine.for %i = 0 to 128 {
        %Aik = affine.load %A[%i, %k] : memref<?x?xf32>
        %Qik = arith.divf %Aik, %Rkk   : f32
        affine.store %Qik, %Q[%i, %k] : memref<?x?xf32>
      }

      // ─────── 3. 处理后续列 j = k+1 … N-1 ───────
      affine.for %j = affine_map<(d0)->(d0 + 1)>(%k) to 128 {
        // 3-a) R[k,j] = Σ_i Q[i,k] * A[i,j]
        %Rkj = affine.for %i = 0 to 128 iter_args(%acc2 = %cst0) -> (f32) {
          %Qik = affine.load %Q[%i, %k] : memref<?x?xf32>
          %Aij = affine.load %A[%i, %j] : memref<?x?xf32>
          %prd = arith.mulf %Qik, %Aij  : f32
          %sum = arith.addf %acc2, %prd : f32
          affine.yield %sum : f32
        }
        affine.store %Rkj, %R[%k, %j] : memref<?x?xf32>

        // 3-b) A[:,j] ← A[:,j] − Q[:,k] * R[k,j]
        affine.for %i = 0 to 128 {
          %Aij_old = affine.load %A[%i, %j] : memref<?x?xf32>
          %Qik     = affine.load %Q[%i, %k] : memref<?x?xf32>
          %Rkj_val = affine.load %R[%k, %j] : memref<?x?xf32>
          %prd2    = arith.mulf %Qik, %Rkj_val : f32
          %Aij_new = arith.subf %Aij_old, %prd2 : f32
          affine.store %Aij_new, %A[%i, %j] : memref<?x?xf32>
        }
      }
    }
    return
  }
}
