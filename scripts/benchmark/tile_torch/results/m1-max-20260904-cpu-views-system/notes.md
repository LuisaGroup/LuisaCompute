# CPU input views versus Torch and direct BLAS, 2026-09-04 UTC

The optimized Tile path remains slower than both libraries on every tested
shape. At 1024³, median times are **5.541 ms Tile, 0.982 ms Torch, and 0.989 ms
Accelerate BLAS**. Paired median Tile/Torch is 5.568× and Tile/BLAS is 5.585×.
Input-copy removal is real progress, not completion of the performance goal.

| M×N×K | Tile µs | Torch µs | BLAS µs | Paired Tile/Torch |
|---|---:|---:|---:|---:|
| 32³ | 4.543 | 0.861 | 0.317 | 5.329 |
| 128³ | 15.703 | 4.863 | 4.193 | 3.196 |
| 512³ | 629.262 | 144.988 | 137.159 | 4.373 |
| 1024³ | 5541.276 | 982.044 | 988.617 | 5.568 |
| 256×1024×128 | 224.925 | 67.263 | 66.765 | 3.313 |
| 1024×128×256 | 181.695 | 64.136 | 62.506 | 2.836 |
| 127×193×61 | 33.366 | 6.512 | 5.901 | 5.130 |
| 513×257×129 | 404.701 | 44.747 | 42.770 | 9.075 |

Each time is the median of six per-round p50s. Ratios are paired round medians,
not quotients of displayed medians. Full paired BLAS ranges and all samples
are retained in [results.md](results.md) and [results.json](results.json).

## Method and interpretation

- Six fresh rounds, all six implementation orders verified for every shape,
  rotating shape order; no historical timings, winner selection, or discarded
  slow rounds. Warmup 200 ms, seven samples × 30 ms, requested eight threads.
- Tile uses fixed 4×16×32 worker tiles, window 2, automatic vectorization,
  8192-byte stack budget, 64 logical pack lanes, and the input-view option.
  The [controlled A/B](../m1-max-20260904-cpu-views-replay/notes.md) isolates
  forwarding from those existing choices. Its raw-source audit also shows
  that the two ragged GEMMs still retain input snapshots.
- Compact row-major FP32, alpha=1, beta=0, no transpose or caller-side
  prepacking. All inputs/outputs stay resident during warm timing. JIT,
  setup, transfers, and cold phases are separately recorded. Warm host-wall
  measurements include API overhead and internal temporary handling.
- Six rounds × eight shapes × three implementations: **144/144 full outputs
  valid**, 33,798,528 checked elements, maximum absolute error zero against
  the full FP64 oracle for the deterministic dyadic inputs. Non-dyadic and
  ordered-math regressions are covered by the C++ suite, not these inputs.
- Torch 2.14.0 reports `BLAS_INFO=accelerate`; the separate system executable
  directly calls classic LP64 `cblas_sgemm`. This identifies APIs/build
  configuration, not an undocumented internal microkernel or instruction set.
  Environment thread counts are requests, not measured worker counts.
- All 20 executable/library paths remained unchanged and were independently
  rehashed after the run. All 48 raw LLVM source hashes were verified.
  Apple M1 Max, macOS 26.6.2; no concurrent build, test, or profiler.

Next work remains structural: broader proved forwarding coverage, cache reuse
across logical programs, useful task granularity, and backend-specific
microkernel/register realization. Cost-model candidates must distinguish a
requested policy from what was actually emitted. A flag that keeps the same
fallback cannot supply causal evidence for the cost of a different layout.

The independent CPU/Metal TIRx and Metal MPP paths remain intact; this CPU run
does not retime the [seven-way MPS/MPP/Torch baseline](../m1-max-20260904-subgroup-sync-lowerings/notes.md).
