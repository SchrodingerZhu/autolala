module {
    func.func @kernel_adi(
        %u: memref<?x?xf32>, 
        %v: memref<?x?xf32>, 
        %p: memref<?x?xf32>, 
        %q: memref<?x?xf32>, 
        %tsteps: index, 
        %n: index
    ) {
        affine.for %loop_once = 0 to 1 {
        // Constants
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c_minus_1 = arith.constant -1 : index
            %c_minus_2 = arith.constant -2 : index
            
            %f0 = arith.constant 0.0 : f32
            %f1 = arith.constant 1.0 : f32
            %f2 = arith.constant 2.0 : f32
            %f_half = arith.constant 0.5 : f32
            
            // Convert indices to f32 for calculations
            %n_f32 = arith.index_cast %n : index to i32
            %n_float = arith.sitofp %n_f32 : i32 to f32
            %tsteps_f32 = arith.index_cast %tsteps : index to i32
            %tsteps_float = arith.sitofp %tsteps_f32 : i32 to f32
            
            // Calculate constants
            %DX = arith.divf %f1, %n_float : f32
            %DY = arith.addf %f0, %DX : f32  // DY = DX (replaced simple assignment)
            %DT = arith.divf %f1, %tsteps_float : f32
            %B1 = arith.addf %f0, %f2 : f32  // B1 = 2.0 (replaced simple assignment)
            %B2 = arith.addf %f0, %f1 : f32  // B2 = 1.0 (replaced simple assignment)
            
            %DX_sq = arith.mulf %DX, %DX : f32
            %DY_sq = arith.mulf %DY, %DY : f32
            %mul1_temp = arith.mulf %B1, %DT : f32
            %mul1 = arith.divf %mul1_temp, %DX_sq : f32
            %mul2_temp = arith.mulf %B2, %DT : f32
            %mul2 = arith.divf %mul2_temp, %DY_sq : f32
            
            %a_temp = arith.divf %mul1, %f2 : f32
            %a = arith.subf %f0, %a_temp : f32  // -mul1/2
            %b = arith.addf %f1, %mul1 : f32
            %c = arith.addf %f0, %a : f32  // c = a (replaced simple assignment)
            %d_temp = arith.divf %mul2, %f2 : f32
            %d = arith.subf %f0, %d_temp : f32  // -mul2/2
            %e = arith.addf %f1, %mul2 : f32
            %f_val = arith.addf %f0, %d : f32  // f = d (replaced simple assignment)
            
            // Calculate loop bounds
            %n_minus_1 = arith.subi %n, %c1 : index
            %n_minus_2 = arith.subi %n, %c2 : index
            %tsteps_plus_1 = arith.addi %tsteps, %c1 : index
            
            // Main time loop: for (t=1; t<=tsteps; t++)
            affine.for %t = 1 to %tsteps_plus_1 {
                
                // Column Sweep
                affine.for %i = 1 to %n_minus_1 {
                    // v[0][i] = 1.0
                    affine.store %f1, %v[0, %i] : memref<?x?xf32>
                    // p[i][0] = 0.0
                    affine.store %f0, %p[%i, 0] : memref<?x?xf32>
                    // q[i][0] = v[0][i]
                    %v_0_i = affine.load %v[0, %i] : memref<?x?xf32>
                    %q_i_0_val = arith.addf %f0, %v_0_i : f32  // q[i][0] = v[0][i] (replaced simple assignment)
                    affine.store %q_i_0_val, %q[%i, 0] : memref<?x?xf32>
                    
                    affine.for %j = 1 to %n_minus_1 {
                        // p[i][j] = -c / (a*p[i][j-1]+b)
                        %j_minus_1 = arith.subi %j, %c1 : index
                        %p_i_j_minus_1 = affine.load %p[%i, %j_minus_1] : memref<?x?xf32>
                        %a_times_p = arith.mulf %a, %p_i_j_minus_1 : f32
                        %denom1 = arith.addf %a_times_p, %b : f32
                        %neg_c = arith.subf %f0, %c : f32
                        %p_i_j_val = arith.divf %neg_c, %denom1 : f32
                        affine.store %p_i_j_val, %p[%i, %j] : memref<?x?xf32>
                        
                        // q[i][j] = (-d*u[j][i-1]+(1.0+2.0*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b)
                        %u_j_i_minus_1 = affine.load %u[%j, %j_minus_1] : memref<?x?xf32>
                        %u_j_i = affine.load %u[%j, %i] : memref<?x?xf32>
                        %u_j_i_plus_1 = affine.load %u[%j, %i+1] : memref<?x?xf32>
                        %q_i_j_minus_1 = affine.load %q[%i, %j_minus_1] : memref<?x?xf32>
                        
                        %d_times_u1 = arith.mulf %d, %u_j_i_minus_1 : f32
                        %neg_d_times_u1 = arith.subf %f0, %d_times_u1 : f32
                        %two_d = arith.mulf %f2, %d : f32
                        %one_plus_two_d = arith.addf %f1, %two_d : f32
                        %middle_term = arith.mulf %one_plus_two_d, %u_j_i : f32
                        %f_times_u3 = arith.mulf %f_val, %u_j_i_plus_1 : f32
                        %a_times_q = arith.mulf %a, %q_i_j_minus_1 : f32
                        
                        %numerator1 = arith.addf %neg_d_times_u1, %middle_term : f32
                        %numerator2 = arith.subf %numerator1, %f_times_u3 : f32
                        %numerator3 = arith.subf %numerator2, %a_times_q : f32
                        %q_i_j_val = arith.divf %numerator3, %denom1 : f32
                        affine.store %q_i_j_val, %q[%i, %j] : memref<?x?xf32>
                    }
                    
                    // v[n-1][i] = 1.0
                    affine.store %f1, %v[%n_minus_1, %i] : memref<?x?xf32>
                    
                    // Backward sweep: for (j=n-2; j>=1; j--)

                    affine.for %j_rev = 2 to %n {
                        %j = arith.subi %n, %j_rev : index
                        // v[j][i] = p[i][j] * v[j+1][i] + q[i][j]
                        %p_i_j_back = affine.load %p[%i, %j] : memref<?x?xf32>
                        %v_j_plus_1_i = affine.load %v[%j+1, %i] : memref<?x?xf32>
                        %q_i_j_back = affine.load %q[%i, %j] : memref<?x?xf32>
                        %prod1 = arith.mulf %p_i_j_back, %v_j_plus_1_i : f32
                        %v_j_i_val = arith.addf %prod1, %q_i_j_back : f32
                        affine.store %v_j_i_val, %v[%j, %i] : memref<?x?xf32>
                    }
                }
                
                // Row Sweep
                affine.for %i = 1 to %n_minus_1 {
                    // u[i][0] = 1.0
                    affine.store %f1, %u[%i, 0] : memref<?x?xf32>
                    // p[i][0] = 0.0
                    affine.store %f0, %p[%i, 0] : memref<?x?xf32>
                    // q[i][0] = u[i][0]
                    %u_i_0 = affine.load %u[%i, 0] : memref<?x?xf32>
                    %q_i_0_val_row = arith.addf %f0, %u_i_0 : f32  // q[i][0] = u[i][0] (replaced simple assignment)
                    affine.store %q_i_0_val_row, %q[%i, 0] : memref<?x?xf32>
                    
                    affine.for %j = 1 to %n_minus_1 {
                        // p[i][j] = -f / (d*p[i][j-1]+e)
                        %j_minus_1 = arith.subi %j, %c1 : index
                        %p_i_j_minus_1_row = affine.load %p[%i, %j_minus_1] : memref<?x?xf32>
                        %d_times_p = arith.mulf %d, %p_i_j_minus_1_row : f32
                        %denom2 = arith.addf %d_times_p, %e : f32
                        %neg_f = arith.subf %f0, %f_val : f32
                        %p_i_j_val_row = arith.divf %neg_f, %denom2 : f32
                        affine.store %p_i_j_val_row, %p[%i, %j] : memref<?x?xf32>
                        
                        // q[i][j] = (-a*v[i-1][j]+(1.0+2.0*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e)
                        %i_minus_1 = arith.subi %i, %c1 : index
                        %v_i_minus_1_j = affine.load %v[%i_minus_1, %j] : memref<?x?xf32>
                        %v_i_j = affine.load %v[%i, %j] : memref<?x?xf32>
                        %v_i_plus_1_j = affine.load %v[%i+1, %j] : memref<?x?xf32>
                        %q_i_j_minus_1_row = affine.load %q[%i, %j_minus_1] : memref<?x?xf32>
                        
                        %a_times_v1 = arith.mulf %a, %v_i_minus_1_j : f32
                        %neg_a_times_v1 = arith.subf %f0, %a_times_v1 : f32
                        %two_a = arith.mulf %f2, %a : f32
                        %one_plus_two_a = arith.addf %f1, %two_a : f32
                        %middle_term_row = arith.mulf %one_plus_two_a, %v_i_j : f32
                        %c_times_v3 = arith.mulf %c, %v_i_plus_1_j : f32
                        %d_times_q = arith.mulf %d, %q_i_j_minus_1_row : f32
                        
                        %numerator1_row = arith.addf %neg_a_times_v1, %middle_term_row : f32
                        %numerator2_row = arith.subf %numerator1_row, %c_times_v3 : f32
                        %numerator3_row = arith.subf %numerator2_row, %d_times_q : f32
                        %q_i_j_val_row = arith.divf %numerator3_row, %denom2 : f32
                        affine.store %q_i_j_val_row, %q[%i, %j] : memref<?x?xf32>
                    }
                    
                    // u[i][n-1] = 1.0
                    affine.store %f1, %u[%i, %n_minus_1] : memref<?x?xf32>
                    
                    // Backward sweep: for (j=n-2; j>=1; j--)
                    affine.for %j_rev = 2 to %n {
                            // u[i][j] = p[i][j] * u[i][j+1] + q[i][j]
                        %j = arith.subi %n, %j_rev : index
                        %p_i_j_back_row = affine.load %p[%i, %j] : memref<?x?xf32>
                        %u_i_j_plus_1 = affine.load %u[%i, %j+1] : memref<?x?xf32>
                        %q_i_j_back_row = affine.load %q[%i, %j] : memref<?x?xf32>
                        %prod2 = arith.mulf %p_i_j_back_row, %u_i_j_plus_1 : f32
                        %u_i_j_val = arith.addf %prod2, %q_i_j_back_row : f32
                        affine.store %u_i_j_val, %u[%i, %j] : memref<?x?xf32>
                    }
                }
            } 
        }
        return
    }
}
