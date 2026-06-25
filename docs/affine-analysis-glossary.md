# Affine Analysis Glossary & Interpretation Guide

The shared knowledge base for reading AutoLALA / `dmd-core` locality analysis.
Every skill under `.claude/skills/` references this file. It defines the terms,
explains the JSON the tools return, and lists the model's known limits so an
agent can set expectations instead of guessing.

The analysis is **symbolic**: metrics come back as algebraic expressions in the
program's named parameters (`M`, `N`, `K`, …), not concrete numbers. That is the
point — it shows how locality *scales*, not just one measured size.

---

## 1. Core quantities

| Term | Meaning |
| --- | --- |
| **Reference / access** | One `read`, `write`, or `update` of one array element in one loop iteration. |
| **Reuse Interval (RI)** | For a given memory access, the number of *accesses executed* between this access and the **next** access to the same data (block). A property of the access stream's *time* axis. |
| **Reuse Distance (RD)** | For a given access, the number of *distinct data blocks* touched between this access and the next reuse of the same block — i.e. the stack/LRU distance. RD is what decides whether reuse survives a cache of a given size. |
| **Data-Movement Distance (DMD)** | The headline metric: the expected memory traffic, modeled as a sum over RD regions of `multiplicity × sqrt(reuse_distance)` (roughly, cost grows with the square root of the footprint that must be retained). Lower DMD ⇒ better locality. |
| **CRI (Concurrent Reuse Interval)** | The parallel analogue of RI: the reuse interval seen once `T` threads interleave their iterations under a schedule. |
| **Compulsory (cold) traffic** | Accesses that are first-touches of their block — unavoidable misses. Reported as `compulsory_accesses_plain`. |
| **Warm (reused) traffic** | Accesses that hit data brought in earlier — the locality you can improve. `warm_accesses_plain`. |
| **Total accesses** | `total_accesses_plain = warm + compulsory`. The size of the access stream. |

### Cache-model knobs (analysis options)

| Option | CLI flag | MCP arg | Default | Meaning |
| --- | --- | --- | --- | --- |
| Block size | `--block-size` | `block_size` | `1` | Elements per cache line/block. Spatial locality: indices in the same block share a line. `1` = element-granularity (pure temporal locality). |
| Num sets | `--num-sets` | `num_sets` | `1` | Number of cache sets modeled. Widen to study set-associative conflict behavior. |
| Max operations | `--max-operations` | `max_operations` | `5_000_000` | Barvinok symbolic-operation budget. Raise if analysis aborts on a large nest; lower to fail fast. |
| Approximation | `--approximation-method` | (forced) | `scale` | The only method. `scale` keeps the leading scaling behavior so named dims (`M`,`N`,`K`) stay visible in the formula and drops lower-order noise. |

---

## 2. Reading the `AnalysisReport` JSON

Returned by `analyze_dsl` / `analyze_mlir` under `report`, and serialized by
`dmd-core`. Every metric has a `_plain` (ASCII) and `_latex` form; **prefer
`_plain` for reasoning and quoting.**

Top-level fields:

| Field | Type | What it tells you |
| --- | --- | --- |
| `options` | object | The `AnalysisOptions` actually used (echo of block_size/num_sets/…). |
| `total_accesses_plain` | string | Symbolic count of all accesses. |
| `warm_accesses_plain` | string | Reused accesses (the improvable part). |
| `compulsory_accesses_plain` | string | Cold first-touch accesses (the floor). |
| `timestamp_space` | string | The iteration/time domain the analysis ran over. |
| `access_map` | string | The access relation (iteration → data) it built. |
| `ri_distribution` | `DistributionEntry[]` | How reuse **intervals** are distributed across the iteration space. |
| `rd_distribution` | `DistributionEntry[]` | How reuse **distances** are distributed — the locality fingerprint. |
| `dmd_terms` | `DmdTerm[]` | The per-region contributions summed into the DMD formula. |
| `dmd_formula_plain` | string | **The headline.** The full symbolic DMD expression. |
| `parallel` | `ParallelAnalysis?` | Present iff the kernel has a `parallel(T) for`. CRI model instead of plain RD/DMD. |
| `notes` | string[] | Caveats the analyzer emitted (approximations, dropped terms). Always read these. |

### `DistributionEntry` (entries of `ri_distribution` / `rd_distribution`)

```
value_plain : string      // the RI or RD value for this region
regions     : DistributionRegion[]
  domain_plain : string   // the iteration sub-domain with this value
  count_plain  : string   // how many accesses fall in it (symbolic)
```

Read an RD distribution as: *"over iteration region `domain_plain`, `count_plain`
accesses have reuse distance `value_plain`."* A small RD with a large count is
good locality; a large RD scaling with `N` means reuse that a finite cache loses.

### `DmdTerm` (entries of `dmd_terms`)

```
domain_plain          : string   // iteration sub-domain
multiplicity_plain    : string   // how many times this cost is paid
reuse_distance_plain  : string   // the RD driving this term
term_plain            : string   // this region's contribution to DMD
```

The DMD formula is the sum of the `term_plain`s. To find the bottleneck, find the
term whose `reuse_distance_plain` grows fastest in the named parameters.

### `ParallelAnalysis` (the `parallel` field)

```
thread_count : i64         // T from parallel(T)
schedule     : string      // e.g. round-robin
cri_entries  : ParallelCriEntry[]
  domain_plain      : string
  source_ri_plain   : string         // the sequential RI this CRI derives from
  model_kind        : "negative_binomial" | "racetrack"
  law               : ParallelCriLaw  // range/cri/probability/count (_plain + _latex)
notes        : string[]
```

`model_kind` names the probabilistic RI law used when `T` threads interleave:
**racetrack** and **negative_binomial** are the two closed forms the model knows.

---

## 3. Interpreting results — a checklist

1. **Does it scale?** Look at `dmd_formula_plain`. If named dims (`M`,`N`,`K`)
   survive, locality depends on problem size — the interesting case. A pure
   constant means a fully-cached working set.
2. **Where is the traffic?** Compare `warm` vs `compulsory`. Mostly compulsory ⇒
   little reuse to exploit (maybe fuse with a producer/consumer). Mostly warm ⇒
   reuse exists; check whether RD is small enough to be captured.
3. **Find the dominant RD region.** In `rd_distribution`, the entry whose
   `value_plain` grows fastest with a large `count` is the locality bottleneck.
   That is the loop whose reuse a real cache will miss.
4. **Pick a transformation hypothesis** (see the transform-advisor skill): a large
   RD that scales with an *outer* loop's trip count usually means **tiling** that
   loop will cut RD to the tile size; a reuse carried by the *wrong* innermost
   loop suggests **interchange**; producer→consumer arrays suggest **fusion**.
5. **Re-evaluate, don't assert.** Always confirm a hypothesis with
   `compare_variants` — symbolic intuition is often right about direction and
   wrong about magnitude.
6. **Read `notes`.** They flag where `scale` dropped terms or where an
   approximation applies.

---

## 4. The DSL (what the analyzer actually consumes)

Grammar (from `crates/dmd-core/src/grammar.lalrpop`):

```
params M, N, K;             // symbolic parameters (declare before use)
array A[M, N];              // array with symbolic extents (rank = #extents)
array C[M, N];

for i in 0 .. M {           // half-open range [lower, upper)
  for j in 0 .. N step 1 {  // optional `step <int>` (default 1)
    if i < j {              // optional guard, conditions joined with &&
      read  A[i, j];        // load
      write C[i, j];        // store
      update C[i, j];       // read-modify-write (load + store of same cell)
    }
  }
}

parallel(8) for p in 0 .. M { ... }   // at most ONE parallel loop per program
```

- **Statements:** `read`, `write`, `update` (`update` = combined load+store).
- **Expressions:** `+`, `-`, `*`, `/` (floor division), unary `-`, parentheses,
  integer literals, and parameter/loop-variable names. Affine forms only.
- **No** `mod`, **no** `ceildiv`, **no** non-constant `*` or `/` between two
  variables — those are not expressible (see limits below).

---

## 5. From MLIR (what `extract_from_mlir` accepts)

Tag the loop nest to extract with the `dmd.extract` attribute, then extract.

| MLIR construct | Becomes | Notes |
| --- | --- | --- |
| `affine.for` | `for v in lo .. hi step s` | bounds from affine maps |
| `affine.if` | `if c0 && c1 { } else { }` | integer-set conditions |
| `affine.load` | `read A[..]` | |
| `affine.store` | `write A[..]` | **store ⇒ write, not update** — fix by hand if it is a read-modify-write |
| `arith.* / math.* / complex.*` | (ignored) | pure compute carries no footprint |

Anything else (e.g. `memref.load`, `scf.for`) is **rejected** with a source span
pointing at the offending op. Common fixes the diagnostics suggest:

- `mod` in an index → rewrite without modulo / extract a tiled form that avoids it.
- `ceildiv` → only floor division (`/`) is supported.
- `memref.load`/`memref.store` → raise to `affine.load`/`affine.store` first.
- `scf.for` / generic loop → convert to `affine.for`.

---

## 6. Known limits (state these up front)

- **One parallel loop** per program; nested/multiple `parallel(T)` is unsupported.
- **`mod` and `ceildiv`** are not expressible; neither is non-affine `*`/`/`
  between two variables.
- **Dynamic memref extents become *independent* parameters** (`A_d0`, `A_d1`, …):
  the analysis cannot know two dynamic dims are equal, so it treats them as free.
- **`affine.store` lowers to `write`, never `update`.** If the MLIR op is truly a
  read-modify-write, change the extracted `write` to `update` so the reuse of that
  cell is counted.
- **Symbolic cost.** Barvinok is serialized (one analysis at a time) and can be
  slow; bound it with `max_operations` and prefer one kernel per call.
- **Metrics are symbolic and approximate** (`scale`): compare variants
  qualitatively by direction and dominant term, not by exact constants.
