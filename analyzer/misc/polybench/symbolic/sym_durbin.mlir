module {
    func.func @kernel_durbin(%r: memref<?xf64>, %y: memref<?xf64>, %n: index) {
        affine.for %loop_once = 0 to 1 {
            %c0 = arith.constant 0.0 : f64
            %c1 = arith.constant 1.0 : f64
            %c0_index = arith.constant 0 : index
            %c1_index = arith.constant 1 : index
            
            // Allocate local arrays
            %z = memref.alloca(%n) : memref<?xf64>
            %alpha_alloc = memref.alloca() : memref<1xf64>
            %beta_alloc = memref.alloca() : memref<1xf64>
            %sum_alloc = memref.alloca() : memref<1xf64>
            
            // y[0] = -r[0];
            %r_0 = affine.load %r[0] : memref<?xf64>
            %neg_r_0 = arith.subf %c0, %r_0 : f64
            affine.store %neg_r_0, %y[0] : memref<?xf64>
            
            // beta = 1.0;
            affine.store %c1, %beta_alloc[0] : memref<1xf64>
            
            // alpha = -r[0];
            affine.store %neg_r_0, %alpha_alloc[0] : memref<1xf64>
            
            // for (k = 1; k < n; k++)
            affine.for %k = 1 to %n {
                // beta = (1-alpha*alpha)*beta;
                %alpha = affine.load %alpha_alloc[0] : memref<1xf64>
                %beta = affine.load %beta_alloc[0] : memref<1xf64>
                %alpha_squared = arith.mulf %alpha, %alpha : f64
                %one_minus_alpha_sq = arith.subf %c1, %alpha_squared : f64
                %new_beta = arith.mulf %one_minus_alpha_sq, %beta : f64
                affine.store %new_beta, %beta_alloc[0] : memref<1xf64>
                
                // sum = 0.0;
                affine.store %c0, %sum_alloc[0] : memref<1xf64>
                
                // for (i=0; i<k; i++) - using affine_map for dependent loop
                affine.for %i = 0 to affine_map<(d0) -> (d0)> (%k) {
                    // sum += r[k-i-1]*y[i];
                    %sum = affine.load %sum_alloc[0] : memref<1xf64>
                    %k_minus_i = arith.subi %k, %i : index
                    %k_minus_i_minus_1 = arith.subi %k_minus_i, %c1_index : index
                    %r_val = memref.load %r[%k_minus_i_minus_1] : memref<?xf64>
                    %y_i = affine.load %y[%i] : memref<?xf64>
                    %prod = arith.mulf %r_val, %y_i : f64
                    %new_sum = arith.addf %sum, %prod : f64
                    affine.store %new_sum, %sum_alloc[0] : memref<1xf64>
                }
                
                // alpha = - (r[k] + sum)/beta;
                %r_k = affine.load %r[%k] : memref<?xf64>
                %sum_final = affine.load %sum_alloc[0] : memref<1xf64>
                %beta_final = affine.load %beta_alloc[0] : memref<1xf64>
                %r_plus_sum = arith.addf %r_k, %sum_final : f64
                %div_result = arith.divf %r_plus_sum, %beta_final : f64
                %new_alpha = arith.subf %c0, %div_result : f64
                affine.store %new_alpha, %alpha_alloc[0] : memref<1xf64>
                
                // for (i=0; i<k; i++) - using affine_map for dependent loop
                affine.for %i = 0 to affine_map<(d0) -> (d0)> (%k) {
                    // z[i] = y[i] + alpha*y[k-i-1];
                    %y_i = affine.load %y[%i] : memref<?xf64>
                    %alpha_curr = affine.load %alpha_alloc[0] : memref<1xf64>
                    %k_minus_i = arith.subi %k, %i : index
                    %k_minus_i_minus_1 = arith.subi %k_minus_i, %c1_index : index
                    %y_k_minus_i_minus_1 = memref.load %y[%k_minus_i_minus_1] : memref<?xf64>
                    %alpha_times_y = arith.mulf %alpha_curr, %y_k_minus_i_minus_1 : f64
                    %z_i = arith.addf %y_i, %alpha_times_y : f64
                    affine.store %z_i, %z[%i] : memref<?xf64>
                }
                
                // for (i=0; i<k; i++) - using affine_map for dependent loop
                affine.for %i = 0 to affine_map<(d0) -> (d0)> (%k) {
                    // y[i] = z[i];
                    %z_i = affine.load %z[%i] : memref<?xf64>
                    affine.store %z_i, %y[%i] : memref<?xf64>
                }
                
                // y[k] = alpha;
                %alpha_final = affine.load %alpha_alloc[0] : memref<1xf64>
                affine.store %alpha_final, %y[%k] : memref<?xf64>
            } 
        }
        return
    }
}
