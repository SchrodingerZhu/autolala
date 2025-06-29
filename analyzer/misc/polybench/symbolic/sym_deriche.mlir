module {
    func.func @kernel_deriche(%imgIn: memref<?x?xf64>, %imgOut: memref<?x?xf64>, %y1: memref<?x?xf64>, %y2: memref<?x?xf64>, %w: index, %h: index) {
        affine.for %loop_once = 0 to 1 {
            %alpha = arith.constant 1.0 : f64
            %c0 = arith.constant 0.0 : f64
            %c1 = arith.constant 1.0 : f64
            %c2 = arith.constant 2.0 : f64
            %c1_index = arith.constant 1 : index
            
            // Calculate coefficients
            %neg_alpha = arith.negf %alpha : f64
            %exp_neg_alpha = math.exp %neg_alpha : f64
            %one_minus_exp = arith.subf %c1, %exp_neg_alpha : f64
            %numerator = arith.mulf %one_minus_exp, %one_minus_exp : f64
            
            %two_alpha = arith.mulf %c2, %alpha : f64
            %two_alpha_exp = arith.mulf %two_alpha, %exp_neg_alpha : f64
            %neg_two_alpha = arith.negf %two_alpha : f64
            %exp_neg_two_alpha = math.exp %neg_two_alpha : f64
            %denom_part = arith.addf %c1, %two_alpha_exp : f64
            %denominator = arith.subf %denom_part, %exp_neg_two_alpha : f64
            %k = arith.divf %numerator, %denominator : f64
            
            %a1 = %k : f64
            %a5 = %k : f64
            %alpha_minus_one = arith.subf %alpha, %c1 : f64
            %k_exp_alpha_minus_one = arith.mulf %k, %exp_neg_alpha : f64
            %a2 = arith.mulf %k_exp_alpha_minus_one, %alpha_minus_one : f64
            %a6 = %a2 : f64
            %alpha_plus_one = arith.addf %alpha, %c1 : f64
            %a3 = arith.mulf %k_exp_alpha_minus_one, %alpha_plus_one : f64
            %a7 = %a3 : f64
            %neg_k_exp_two_alpha = arith.negf %k : f64
            %a4 = arith.mulf %neg_k_exp_two_alpha, %exp_neg_two_alpha : f64
            %a8 = %a4 : f64
            
            %b1 = math.powf %c2, %neg_alpha : f64
            %b2 = arith.negf %exp_neg_two_alpha : f64
            %c1_coeff = %c1 : f64
            %c2_coeff = %c1 : f64
            
            // Allocate temporary variables
            %ym1_alloc = memref.alloca() : memref<1xf64>
            %ym2_alloc = memref.alloca() : memref<1xf64>
            %xm1_alloc = memref.alloca() : memref<1xf64>
            %yp1_alloc = memref.alloca() : memref<1xf64>
            %yp2_alloc = memref.alloca() : memref<1xf64>
            %xp1_alloc = memref.alloca() : memref<1xf64>
            %xp2_alloc = memref.alloca() : memref<1xf64>
            %tm1_alloc = memref.alloca() : memref<1xf64>
            %tp1_alloc = memref.alloca() : memref<1xf64>
            %tp2_alloc = memref.alloca() : memref<1xf64>
            
            // First loop: forward pass in j direction
            affine.for %i = 0 to %w {
                affine.store %c0, %ym1_alloc[0] : memref<1xf64>
                affine.store %c0, %ym2_alloc[0] : memref<1xf64>
                affine.store %c0, %xm1_alloc[0] : memref<1xf64>
                
                affine.for %j = 0 to %h {
                    %imgIn_ij = affine.load %imgIn[%i, %j] : memref<?x?xf64>
                    %ym1 = affine.load %ym1_alloc[0] : memref<1xf64>
                    %ym2 = affine.load %ym2_alloc[0] : memref<1xf64>
                    %xm1 = affine.load %xm1_alloc[0] : memref<1xf64>
                    
                    %term1 = arith.mulf %a1, %imgIn_ij : f64
                    %term2 = arith.mulf %a2, %xm1 : f64
                    %term3 = arith.mulf %b1, %ym1 : f64
                    %term4 = arith.mulf %b2, %ym2 : f64
                    %sum1 = arith.addf %term1, %term2 : f64
                    %sum2 = arith.addf %term3, %term4 : f64
                    %y1_ij = arith.addf %sum1, %sum2 : f64
                    affine.store %y1_ij, %y1[%i, %j] : memref<?x?xf64>
                    
                    affine.store %imgIn_ij, %xm1_alloc[0] : memref<1xf64>
                    affine.store %ym1, %ym2_alloc[0] : memref<1xf64>
                    affine.store %y1_ij, %ym1_alloc[0] : memref<1xf64>
                }
            }
            
            // Second loop: backward pass in j direction
            affine.for %i = 0 to %w {
                affine.store %c0, %yp1_alloc[0] : memref<1xf64>
                affine.store %c0, %yp2_alloc[0] : memref<1xf64>
                affine.store %c0, %xp1_alloc[0] : memref<1xf64>
                affine.store %c0, %xp2_alloc[0] : memref<1xf64>
                
                %h_minus_1 = arith.subi %h, %c1_index : index
                affine.for %j = affine_map<(d0) -> (d0)> (%h_minus_1) to affine_map<() -> (-1)> () step -1 {
                    %yp1 = affine.load %yp1_alloc[0] : memref<1xf64>
                    %yp2 = affine.load %yp2_alloc[0] : memref<1xf64>
                    %xp1 = affine.load %xp1_alloc[0] : memref<1xf64>
                    %xp2 = affine.load %xp2_alloc[0] : memref<1xf64>
                    
                    %term1 = arith.mulf %a3, %xp1 : f64
                    %term2 = arith.mulf %a4, %xp2 : f64
                    %term3 = arith.mulf %b1, %yp1 : f64
                    %term4 = arith.mulf %b2, %yp2 : f64
                    %sum1 = arith.addf %term1, %term2 : f64
                    %sum2 = arith.addf %term3, %term4 : f64
                    %y2_ij = arith.addf %sum1, %sum2 : f64
                    affine.store %y2_ij, %y2[%i, %j] : memref<?x?xf64>
                    
                    %imgIn_ij = affine.load %imgIn[%i, %j] : memref<?x?xf64>
                    affine.store %xp1, %xp2_alloc[0] : memref<1xf64>
                    affine.store %imgIn_ij, %xp1_alloc[0] : memref<1xf64>
                    affine.store %yp1, %yp2_alloc[0] : memref<1xf64>
                    affine.store %y2_ij, %yp1_alloc[0] : memref<1xf64>
                }
            }
            
            // Third loop: combine y1 and y2
            affine.for %i = 0 to %w {
                affine.for %j = 0 to %h {
                    %y1_ij = affine.load %y1[%i, %j] : memref<?x?xf64>
                    %y2_ij = affine.load %y2[%i, %j] : memref<?x?xf64>
                    %sum = arith.addf %y1_ij, %y2_ij : f64
                    %result = arith.mulf %c1_coeff, %sum : f64
                    affine.store %result, %imgOut[%i, %j] : memref<?x?xf64>
                }
            }
            
            // Fourth loop: forward pass in i direction
            affine.for %j = 0 to %h {
                affine.store %c0, %tm1_alloc[0] : memref<1xf64>
                affine.store %c0, %ym1_alloc[0] : memref<1xf64>
                affine.store %c0, %ym2_alloc[0] : memref<1xf64>
                
                affine.for %i = 0 to %w {
                    %imgOut_ij = affine.load %imgOut[%i, %j] : memref<?x?xf64>
                    %tm1 = affine.load %tm1_alloc[0] : memref<1xf64>
                    %ym1 = affine.load %ym1_alloc[0] : memref<1xf64>
                    %ym2 = affine.load %ym2_alloc[0] : memref<1xf64>
                    
                    %term1 = arith.mulf %a5, %imgOut_ij : f64
                    %term2 = arith.mulf %a6, %tm1 : f64
                    %term3 = arith.mulf %b1, %ym1 : f64
                    %term4 = arith.mulf %b2, %ym2 : f64
                    %sum1 = arith.addf %term1, %term2 : f64
                    %sum2 = arith.addf %term3, %term4 : f64
                    %y1_ij = arith.addf %sum1, %sum2 : f64
                    affine.store %y1_ij, %y1[%i, %j] : memref<?x?xf64>
                    
                    affine.store %imgOut_ij, %tm1_alloc[0] : memref<1xf64>
                    affine.store %ym1, %ym2_alloc[0] : memref<1xf64>
                    affine.store %y1_ij, %ym1_alloc[0] : memref<1xf64>
                }
            }
            
            // Fifth loop: backward pass in i direction
            affine.for %j = 0 to %h {
                affine.store %c0, %tp1_alloc[0] : memref<1xf64>
                affine.store %c0, %tp2_alloc[0] : memref<1xf64>
                affine.store %c0, %yp1_alloc[0] : memref<1xf64>
                affine.store %c0, %yp2_alloc[0] : memref<1xf64>
                
                %w_minus_1 = arith.subi %w, %c1_index : index
                affine.for %i = affine_map<(d0) -> (d0)> (%w_minus_1) to affine_map<() -> (-1)> () step -1 {
                    %tp1 = affine.load %tp1_alloc[0] : memref<1xf64>
                    %tp2 = affine.load %tp2_alloc[0] : memref<1xf64>
                    %yp1 = affine.load %yp1_alloc[0] : memref<1xf64>
                    %yp2 = affine.load %yp2_alloc[0] : memref<1xf64>
                    
                    %term1 = arith.mulf %a7, %tp1 : f64
                    %term2 = arith.mulf %a8, %tp2 : f64
                    %term3 = arith.mulf %b1, %yp1 : f64
                    %term4 = arith.mulf %b2, %yp2 : f64
                    %sum1 = arith.addf %term1, %term2 : f64
                    %sum2 = arith.addf %term3, %term4 : f64
                    %y2_ij = arith.addf %sum1, %sum2 : f64
                    affine.store %y2_ij, %y2[%i, %j] : memref<?x?xf64>
                    
                    %imgOut_ij = affine.load %imgOut[%i, %j] : memref<?x?xf64>
                    affine.store %tp1, %tp2_alloc[0] : memref<1xf64>
                    affine.store %imgOut_ij, %tp1_alloc[0] : memref<1xf64>
                    affine.store %yp1, %yp2_alloc[0] : memref<1xf64>
                    affine.store %y2_ij, %yp1_alloc[0] : memref<1xf64>
                }
            }
            
            // Sixth loop: final combination
            affine.for %i = 0 to %w {
                affine.for %j = 0 to %h {
                    %y1_ij = affine.load %y1[%i, %j] : memref<?x?xf64>
                    %y2_ij = affine.load %y2[%i, %j] : memref<?x?xf64>
                    %sum = arith.addf %y1_ij, %y2_ij : f64
                    %result = arith.mulf %c2_coeff, %sum : f64
                    affine.store %result, %imgOut[%i, %j] : memref<?x?xf64>
                }
            } 
        }
        return
    }
}
