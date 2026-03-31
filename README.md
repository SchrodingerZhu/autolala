# Automatical Loop Algebraic Locality Analysis (AutoLALA)

`autodmd` is a Rust workspace for symbolic data movement complexity analysis of affine loop-tree programs. It replaces the MLIR ingestion path with a small DSL, lowers programs into polyhedral sets/maps with `barvinok-rs`, computes reuse-interval and reuse-distance distributions, and builds symbolic DMD formulas as sums of square roots over reuse-distance regions.

The analysis runs Barvinok with `--approximation-method=scale` and filters non-scaling DMD branches whose symbolic domains pin or upper-bound named scaling dimensions such as `K = 2`.

## Workspace

- `crates/dmd-core`: DSL parser, semantic checks, polyhedral lowering, RI/RD analysis, formula rendering.
- `crates/dmd-cli`: command-line frontend for file or stdin analysis.
- `crates/dmd-playground`: Axum-based playground with a Monaco editor, KaTeX rendering, bounded task execution, and polling APIs.

## DSL Overview

The DSL supports:

- symbolic parameters via `params`
- affine array declarations via `array`
- affine `for` loops with optional `step`
- affine `if` guards with `&&`
- `read`, `write`, and `update` accesses

Example:

```text
params M, N, K;
array A[M, K];
array B[K, N];
array C[M, N];

for i in 0 .. M {
  for j in 0 .. N {
    for k in 0 .. K {
      read C[i, j];
      read A[i, k];
      read B[k, j];
      write C[i, j];
    }
  }
}
```

## Native Dependencies

`autodmd` uses the pinned `barvinok-rs` revision from the workspace `Cargo.toml`. Its `barvinok-sys` build script bundles Barvinok, isl, and PolyLib sources, but it still requires:

- autotools to reconfigure the bundled C sources
- a C/C++ toolchain
- `libclang` for `bindgen`
- dynamic GMP and NTL libraries

### Linux

On Ubuntu or Debian, install:

```bash
sudo apt-get update
sudo apt-get install -y \
  autoconf \
  automake \
  libtool \
  pkg-config \
  build-essential \
  clang \
  libclang-dev \
  libgmp-dev \
  libntl-dev
```

No extra prefix environment variables are normally required on Linux.

### macOS

On macOS, `barvinok-sys/build.rs` expects Homebrew-style GMP and NTL prefixes. Install:

```bash
brew install autoconf automake libtool pkg-config llvm gmp ntl
```

Then export:

```bash
export GMP_PREFIX="$(brew --prefix gmp)"
export NTL_PREFIX="$(brew --prefix ntl)"
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
```

## Build And Test

From the repository root:

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

If your shell injects a `RUSTC_WRAPPER` that causes local issues, unset it for the command you run.

## CLI

Analyze a file:

```bash
cargo run -p dmd-cli -- --input kernel.dsl
```

Read from stdin:

```bash
cat kernel.dsl | cargo run -p dmd-cli --
```

Emit JSON:

```bash
cargo run -p dmd-cli -- --input kernel.dsl --json
```

Useful flags:

- `--block-size`
- `--num-sets`
- `--max-operations`
- `--approximation-method scale`
- `--json`

## Playground

Start the local playground:

```bash
cargo run -p dmd-playground -- --port 3000
```

Then open `http://127.0.0.1:3000`.

The playground provides:

- a Monaco-backed DSL editor with custom highlighting
- KaTeX rendering for formulas
- bounded task concurrency and payload sizes
- timeout-guarded symbolic analysis jobs

The backend API is:

- `POST /api/tasks`
- `GET /api/tasks/{task_id}`

## Notes

- The Barvinok context is initialized with `--approximation-method=scale`.
- DMD aggregation excludes special-case branches that do not scale asymptotically.
- RI and RD distributions are still reported in full for diagnostics.
- In-process Barvinok initialization is serialized because the current `Context::from_args(...)` path is not stable under parallel in-process calls.

## CI

GitHub Actions runs on Ubuntu, installs the Barvinok native prerequisites, checks formatting, validates the playground JavaScript with `node --check`, runs the workspace tests, and enforces strict Clippy.

## License

Licensed under `MIT OR Apache-2.0`.
