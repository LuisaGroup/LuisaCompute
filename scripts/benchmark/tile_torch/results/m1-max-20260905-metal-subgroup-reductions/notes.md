# Metal SIMD-group reduction and normalization report

Date: 2026-09-05 Asia/Shanghai (`2026-09-04T23:14:31.938143Z` in the
raw report). Machine: Apple M1 Max, macOS 26.6.2 arm64. PyTorch 2.14.0.
Source revision: `474c28dfdbfa00a7c32eb007956256c0ba0f1a88` plus the recorded
uncommitted Tile/TIRx implementation.

## Result

The opt-in TileIR→TIRx Metal SIMD-group reduction realization passes the
complete FP64 oracle for all 12 measured FP32 cases. On this cohort its
synchronized device-resident host-wall throughput is faster than eager
PyTorch MPS in all 12 cases: Tile/Torch ranges from 0.124× to 0.902×.
The sum and softmax comparisons use preallocated outputs on both sides. The
RMSNorm comparison is less symmetric because PyTorch's public functional API
allocates its returned output inside the timed region; the raw report marks
that policy per row.

This is evidence for row sum, softmax and RMSNorm at the four listed shapes.
It is not a claim of production LLM-suite, other-device, other-dtype, pure
GPU-event, or arbitrary reduction parity.

| Operator / rows×width | Tile µs | Torch µs | Tile/Torch | Threads | SIMD groups/program | Private stripe/worker |
|---|---:|---:|---:|---:|---:|---:|
| sum 1×127 | 3.268 | 7.211 | 0.453× | 32 | 1 | 0 |
| sum 17×257 | 3.106 | 4.340 | 0.716× | 256 | 1 | 0 |
| sum 128×1024 | 3.387 | 5.604 | 0.604× | 128 | 4 | 0 |
| sum 64×4096 | 4.721 | 16.119 | 0.293× | 256 | 8 | 0 |
| softmax 1×127 | 3.578 | 26.111 | 0.137× | 64 | 2 | 2 |
| softmax 17×257 | 3.305 | 26.594 | 0.124× | 256 | 1 | 9 |
| softmax 128×1024 | 5.385 | 30.376 | 0.177× | 128 | 4 | 8 |
| softmax 64×4096 | 8.881 | 31.029 | 0.286× | 256 | 8 | 16 |
| RMSNorm 1×127 | 3.904 | 7.155 | 0.546× | 64 | 2 | 0 |
| RMSNorm 17×257 | 5.335 | 6.154 | 0.867× | 256 | 1 | 0 |
| RMSNorm 128×1024 | 6.673 | 8.707 | 0.766× | 128 | 4 | 0 |
| RMSNorm 64×4096 | 11.177 | 12.392 | 0.902× | 256 | 8 | 0 |

Times are p50s across 11 calibrated samples after warmup. Ratio is
Tile/Torch, so smaller is faster. Each warm sample uses a calibrated dispatch
batch and synchronizes before the host clock stops. Setup allocation, upload,
capture, compilation, first call and download are excluded from warm
throughput and retained separately in `results.json`. Native writes a
preallocated output. Torch sum/softmax use `out=`; Torch's functional RMSNorm
has no `out=` overload, so its returned-output allocation is included in warm
timing and explicitly recorded. This is not Metal GPU counter or command-buffer
event timing.

## What changed structurally

The reference lowering assigned one logical row program to one Metal thread.
Its row reduction remained serial, and softmax could materialize a logical
`float[4096]` in every thread. The new target-local mapper instead:

1. revalidates an exact pure FP32 add/max/min reduction contract;
2. enumerates one, two, four and eight SIMD groups per logical program;
3. packs independent short programs into one threadgroup when one SIMD group
   per program wins;
4. stripes wide rows across cooperating workers and combines uniform SIMD
   collectives through a small shared partial array; and
5. compacts an eligible compiler-owned logical Tile to the worker's owned
   stripe only after proving every flattened access equals the distributed
   element coordinate.

For `softmax 64×4096`, the selected plan uses 256 workers, eight SIMD groups,
16 private FP32 values per worker and two 8-value shared partial arrays. The
generated source therefore has a private `[16]` stripe rather than one
per-thread `[4096]` array. Memory is a consequence of the selected execution
map; it is not treated as an execution hierarchy.

The plan cost basis is `metal_subgroup_reduction_v1`. Its coefficients are an
analytic prior in abstract issue rounds, not nanoseconds. Exact
`--group-threads` candidates and the outer staged/JIT runner remain the
authority for measured selection.

## Numerical and legality contract

`--metal-subgroup-reductions` is both an explicit realization switch and
permission to replace the reference FP32 left fold with a tree reduction.
The pass requires Metal, a 32-lane SIMD width, automatic or subgroup root
binding, noalias buffer arguments, static positive domains, exact identity
values, pure contributions, uniform reduction control and partitioned global
stores. Unsupported effects, nested logical parallel regions, divergent
reductions, unsupported manual memory, malformed annotations or over-capacity
candidates do not enter the solver.

Every output element was checked. Maximum absolute error in the saved run was:

- sum: exactly zero for the deterministic test inputs;
- softmax: at most `1.879e-9`;
- RMSNorm: at most `1.570e-7`.

The tolerances and every raw error value are in `results.json`; failed cases
would remain in the report and make the runner exit nonzero.

## Exact command

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/luisa-reductions-subgroup-metal-20260905-final-reported \
  --backends metal --operations sum,softmax,rmsnorm \
  --metal-subgroup-reductions \
  --samples 11 --sample-ms 100 --warmup-ms 100 --capture-sources
```

No build, profiler or other GPU workload ran concurrently. The output
directory was new. The driver recaptured/JIT-compiled each shape and checked
the requested policy and realized plan before accepting timing.

## Artifact identity

- `results.json` SHA-256:
  `6d075e6f0d5a00c8c53f9026884f936b820f25e801cc369838a5daf8188502d1`
- benchmark executable SHA-256:
  `4c0bd9b660085d8d8cd5b3682c8eda4732e2c55853ec77c943986ebd62ac01dd`
- loaded `libluisa-tile-bridge-tirx.dylib` SHA-256:
  `c0374c27b23e6741751f53e08eec5817b4f535b365f8270cfd200a6271670a21`

The raw report records a dirty worktree because it was intentionally measured
before committing. These content hashes, the complete planner records and the
command identify the measured implementation more precisely than the base Git
revision alone.

## Post-measurement verification

The measured artifact came from a complete configured build. With the
submitted `metal::mem_flags(3)` source restored for verification, the complete
`unit_tile` label passed 31/31 tests in 100.18 seconds and the separately
labeled native Tile Runtime integration passed 1/1. The current benchmark
Python discovery suite passed 67/67. A warnings-as-errors Sphinx build passed
after suppressing only the repository's known missing-Doxygen/tutorial
warnings; the new page, figures and download links introduced no warning. The
user's uncommitted local `mem_flags(2)` edit was restored immediately
afterwards and is not part of this feature or its intended commit.

## Remaining boundary

- The implemented matcher is static FP32 add/max/min on Metal; BF16/FP16,
  variance pairs, arg reductions and custom monoids need typed contracts.
- The `{1,2,4,8}` search and coefficients are M1-class bootstrap priors. They
  have not been calibrated on a held-out device or operator set.
- The report covers three operator families, not LayerNorm, cross-entropy,
  fused attention, Top-K, sort, convolution or end-to-end models.
- The current benchmark measures eager Torch and Runtime/API overhead. A
  separate counter/event harness is needed to attribute pure kernel latency.

The independent balanced reference/candidate replay is in
[`../m1-max-20260905-metal-subgroup-rmsnorm-replay/notes.md`](../m1-max-20260905-metal-subgroup-rmsnorm-replay/notes.md).
