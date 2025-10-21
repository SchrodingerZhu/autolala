# Dimension Padding Applied to Einsum Constant Global Benchmarks

## Summary

All array declarations in the `einsum/constant_global` directory have been modified to pad the last dimension to be a multiple of both 12 and 8 (i.e., a multiple of 24).

## Rationale

The padding ensures that the last dimension of all arrays is compatible with:
- 12-way cache configurations
- 8-way cache configurations
- Any divisor of 24

This is achieved by using LCM(12, 8) = 24 as the padding target.

## Transformation Applied

### Padding Rule
- **Original dimension**: Any value (e.g., 64, 16, 8)
- **Padded dimension**: Next multiple of 24
  - 8 → 24
  - 16 → 24
  - 64 → 72

### Examples

#### Example 1: batch_matmul.c
**Before:**
```c
#define K_SIZE 64
#define J_SIZE 64

volatile DATA_TYPE A[B_SIZE][I_SIZE][K_SIZE];
volatile DATA_TYPE B_mat[B_SIZE][K_SIZE][J_SIZE];
volatile DATA_TYPE C[B_SIZE][I_SIZE][J_SIZE];
```

**After:**
```c
#define K_SIZE 64
#define J_SIZE 64

volatile DATA_TYPE A[B_SIZE][I_SIZE][72];
volatile DATA_TYPE B_mat[B_SIZE][K_SIZE][72];
volatile DATA_TYPE C[B_SIZE][I_SIZE][72];
```

#### Example 2: tucker_decomposition_pattern.c
**Before:**
```c
#define L_SIZE 16
#define D_DIM 16

volatile DATA_TYPE X[I_SIZE][J_SIZE][K_SIZE][L_SIZE];
volatile DATA_TYPE A[A_DIM][I_SIZE];
volatile DATA_TYPE Y[A_DIM][B_DIM][C_DIM][D_DIM];
```

**After:**
```c
#define L_SIZE 16
#define D_DIM 16

volatile DATA_TYPE X[I_SIZE][J_SIZE][K_SIZE][24];
volatile DATA_TYPE A[A_DIM][24];
volatile DATA_TYPE Y[A_DIM][B_DIM][C_DIM][24];
```

#### Example 3: marginalization_pattern.c
**Before:**
```c
#define SMALL_SIZE 8
#define M_SIZE 8

volatile DATA_TYPE A[SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE];
volatile DATA_TYPE result[M_SIZE];
```

**After:**
```c
#define SMALL_SIZE 8
#define M_SIZE 8

volatile DATA_TYPE A[SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][SMALL_SIZE][24];
volatile DATA_TYPE result[24];
```

## Important Notes

1. **#define values remain unchanged**: The `#define` macros (e.g., `K_SIZE`, `J_SIZE`) keep their original values. This ensures loop bounds remain correct.

2. **Only last dimension is padded**: The transformation only affects the last (rightmost) dimension of each array declaration.

3. **Numeric literals used**: The padded dimensions use numeric literals (e.g., `72`, `24`) rather than macro names, making the padding explicit.

4. **Loop bounds unchanged**: Since the loop variables still use the original `#define` constants, the actual computation only accesses the valid (non-padded) portion of the arrays.

5. **Memory layout optimized**: The padding ensures better cache line alignment and reduces conflicts for various cache associativities.

## Automation Script

The transformation was automated using `pad_dimensions.py`, which:
1. Parses all `#define` size constants
2. Identifies all `volatile DATA_TYPE` array declarations
3. Extracts the last dimension of each array
4. Calculates the next multiple of 24
5. Replaces array declarations with padded versions

## Files Transformed

All 16 benchmark files in `einsum/constant_global`:
- batch_matmul.c
- bilinear_transformation_pattern.c
- hadamard_product_2d.c
- hadamard_product_4d.c
- mahalanobis_distance_pattern.c
- marginalization_pattern.c
- matrix_chain_5.c
- matrix_diagonal.c
- matrix_trace.c
- maxcut_quantum_circuit_pattern.c
- tensor_network_2x3_pattern.c
- tensor_regression_network_pattern.c
- triplestore_query_pattern.c
- tucker_decomposition_pattern.c
- vector_outer_product.c
- weighted_model_counting_pattern.c
