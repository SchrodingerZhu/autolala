## Requirements

1. Create a new Cargo workspace in `auto-dmd`.
2. Use the latest `barvinok-rs` upstream revision rather than the older `autolala` lockfile revision.
3. Remove the MLIR dependency path entirely and replace it with a purpose-built DSL for loop trees.
4. Keep the core analysis in a reusable library crate.
5. Provide a CLI crate built on top of the core library.
6. Provide a playground website that:
   - accepts DSL input,
   - runs symbolic analysis concurrently,
   - enforces bounded resources for each task,
   - returns well-formatted formulas and structured diagnostics.
7. Compute symbolic reuse-interval distributions in the style of `autolala`.
8. Compute symbolic data movement complexity from the reuse information using the Barvinok-based polyhedral method.
9. For the data movement stage, avoid Denning recursion. Reuse distance should be derived by counting the image of the iteration space under the access map.
10. Keep the implementation production-oriented: explicit error types, tests, structured modules, documented formats, and operational safeguards in the playground.

## Working Assumptions

1. The current latest `barvinok-rs` head is commit `cc72f5071bb7a1a9ec11e228fcb1686c2f8d3f6c` from 2026-03-30 UTC and should be pinned explicitly for reproducibility.
2. The DSL should cover affine loop nests, affine guards, symbolic bounds, and affine memory accesses. That is enough to replace the current MLIR ingestion path for the target use case.
3. The first production version should focus on affine integer programs with unit-stride or positive constant-stride loops and affine accesses.
4. The playground should be a server-rendered or static-asset-backed Rust web app with a JSON API and a bounded in-process worker model. The sandboxing in this version will be implemented as resource caps, timeouts, concurrency limits, and request-size limits inside the service boundary.

## Designed Architecture

### Workspace

- `crates/dmd-core`
  - DSL AST and parser
  - semantic validation
  - lowering to polyhedral sets/maps
  - RI distribution analysis
  - DMD formula construction
  - formula formatting and serialization
- `crates/dmd-cli`
  - command-line interface
  - file/stdin input
  - text/JSON output
- `crates/dmd-playground`
  - web server
  - HTML/CSS/JS frontend
  - sandboxed concurrent execution service

### Analysis Model

1. Parse the DSL into a loop-tree IR with affine expressions and named arrays.
2. Lower the loop tree into:
   - iteration/timestamp space,
   - access map from timestamps to accessed image points.
3. Compute symbolic reuse intervals using the access map composition pattern used in `autolala`.
4. Compute symbolic reuse distance by counting distinct accessed image points in the relevant predecessor window instead of using timestamp-space counting and without Denning recursion.
5. Build DMD as the symbolic sum of square roots of the reuse-distance expression over the access space or its piecewise regions.
6. Render formulas in both plain text and LaTeX-oriented output for CLI and web presentation.

## Action List

1. Initialize the Cargo workspace and crate layout.
2. Define the DSL grammar and the loop-tree IR.
3. Implement semantic checks and normalized affine forms.
4. Integrate `barvinok-rs` at the pinned latest commit and wrap the required ISL/Barvinok operations behind a narrow adapter layer.
5. Port the relevant RI machinery from `autolala` into DSL-driven lowering.
6. Add the access-image-based reuse-distance and DMD computation path.
7. Build formula printers and JSON result models.
8. Implement the CLI.
9. Implement the playground backend with concurrency limits, timeouts, task IDs, and bounded execution.
10. Implement the playground frontend with polished formula presentation and diagnostics.
11. Build, test, and iterate on integration issues.


## Additional Content
1. Consider a more efficient way to compute the reuse distance under infinite-repeat: wrap the whole space with a additional loop that runs twice, then half the final sum. This should directly gives all the imaginary reuse without the complexity of handling an extra symbol
2. Are you sure you still need to keep lalrpop's lexical definition when using together when logos? lalrpop's default lexer is pretty bad in terms of performance.