# Einsum Programs with Global Memory Layout

This directory contains ported versions of the einsum constant programs, converted to use global memory layout similar to the polybench format.

## Transformations Applied

All programs in `../constant/` have been transformed with the following changes:

1. **Data Type Changed**: `DATA_TYPE` changed from `float` to `double`
2. **Global Memory Layout**: Function parameters converted to global `volatile` arrays
3. **Kernel Signature**: Functions now take no parameters (e.g., `void kernel_name()`)
4. **Literal Updates**: Float literals changed from `.0f` to `.0`
5. **Pointer Handling**: Pointer parameters converted to scalar globals with dereferences removed

## Format Comparison

### Original Format (einsum/constant)
```c
#define DATA_TYPE float

void kernel_example(DATA_TYPE A[M][N], DATA_TYPE B[M][N], DATA_TYPE C[M][N]) {
  // ...
  C[i][j] = 0.0f;
  // ...
}
```

### New Format (einsum/constant_global)
```c
#define DATA_TYPE double

volatile DATA_TYPE A[M][N];
volatile DATA_TYPE B[M][N];
volatile DATA_TYPE C[M][N];

void kernel_example() {
  // ...
  C[i][j] = 0.0;
  // ...
}
```

## Transformation Script

The transformation was automated using `port_to_global.py` in the parent directory.

## File List

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

All files follow the same global memory pattern as `polybench/polygeist/constant/3mm.c`.
