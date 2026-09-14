# Metal subgroup LayerNorm and cross-entropy report

Date: 2026-09-05 Asia/Shanghai (`2026-09-04T23:48:22.159442Z` in the
raw report). Machine: Apple M1 Max, macOS 26.6.2 arm64. PyTorch 2.14.0.
Source revision: `7958d84dd28991ed617b54bf53b5b47039573aa0`.

## Result

All eight FP32 LayerNorm and per-row cross-entropy cases pass complete FP64
oracles. The TileIR→TIRx subgroup realization is faster than eager PyTorch MPS
in every measured row. LayerNorm is 0.51×--0.65× the Torch time. Cross-entropy
is 0.03×--0.05×, but that much larger gap includes PyTorch's general
`functional.cross_entropy` dispatch and returned-output allocation; it is not
a pure kernel comparison.

| Operator / rows×width | Tile µs | Torch µs | Tile/Torch | Threads | SIMD groups/program | Shared bytes |
|---|---:|---:|---:|---:|---:|---:|
| LayerNorm 1×127 | 4.500 | 8.400 | 0.536× | 64 | 2 | 16 |
| LayerNorm 17×257 | 5.714 | 8.821 | 0.648× | 256 | 1 | 0 |
| LayerNorm 128×1024 | 7.542 | 13.726 | 0.549× | 128 | 4 | 32 |
| LayerNorm 64×4096 | 12.413 | 24.313 | 0.511× | 256 | 8 | 64 |
| cross-entropy 1×127 | 4.513 | 107.246 | 0.042× | 32 | 1 | 0 |
| cross-entropy 17×257 | 3.449 | 107.695 | 0.032× | 256 | 1 | 0 |
| cross-entropy 128×1024 | 4.290 | 110.171 | 0.039× | 128 | 4 | 32 |
| cross-entropy 64×4096 | 5.838 | 112.263 | 0.052× | 256 | 8 | 64 |

Times are p50 warm synchronized host-wall throughput across 11 samples with
100 ms calibrated sample windows and 100 ms warmup. Inputs stay resident on
the device. Native writes a preallocated output. The PyTorch functional
LayerNorm and cross-entropy calls return newly allocated tensors, recorded as
`output_policy=framework_return_value`. Capture, compilation, allocation,
upload, first call and download are excluded from the warm table and retained
in `results.json`. These are not GPU counter/event timings.

## Structural repair exercised by cross-entropy

Cross-entropy combines two row reductions with a label-dependent gather:

```text
peak     = reduce_max(logits)
total    = reduce_sum(exp(logits - peak))
selected = logits[label]
loss     = log(total) + peak - selected
```

The first subgroup prototype distributed initialization of a logical
`float[4096]` Tile across 256 workers while leaving one such array private to
each worker. Thread zero then read `private_logits[label]`; unless it owned
that label, the value was uninitialized. Six of seven hardware-test rows
failed. This was an execution-to-resource ownership error, not a reduction
arithmetic error.

The repaired bridge has two independent gates:

1. read-only Tile forwarding now carries the path condition of pure lazy
   `if_then_else` expressions. The gather guard proves the temporary index is
   in range, so the immutable snapshot becomes a guarded direct global Tensor
   read;
2. a general distributed-local audit proves `flatten(index) == owner` for
   every use of any nonscalar private Tile with distributed stores. If that
   proof is unknown, automatic subgroup mapping declines and the reference
   execution is retained.

Generated width-4096 cross-entropy source contains direct guarded
`arg0_ptr[row * 4096 + label]` access and no per-thread
`tile_storage_0[4096]`. A separate regression explicitly materializes a
derived private Tile, making forwarding impossible; the ownership audit then
rejects subgroup realization instead of emitting an invalid cross-worker
read.

## Plans and numerical checks

Both operators contain exactly two proved reductions. LayerNorm also has one
width-sized independent output domain; cross-entropy has three scalar
independent elements after input-view forwarding.

| Case | Threads | Groups/program | Reduction elements | Independent elements | Model score |
|---|---:|---:|---:|---:|---:|
| LayerNorm 1×127 | 64 | 2 | 254 | 127 | 30 |
| LayerNorm 17×257 | 256 | 1 | 514 | 257 | 33 |
| LayerNorm 128×1024 | 128 | 4 | 2048 | 1024 | 56 |
| LayerNorm 64×4096 | 256 | 8 | 8192 | 4096 | 96 |
| cross-entropy 1×127 | 32 | 1 | 254 | 3 | 31 |
| cross-entropy 17×257 | 256 | 1 | 514 | 3 | 27 |
| cross-entropy 128×1024 | 128 | 4 | 2048 | 3 | 51 |
| cross-entropy 64×4096 | 256 | 8 | 8192 | 3 | 83 |

Maximum native absolute error is `2.913e-7` for LayerNorm and `1.109e-6`
for cross-entropy. LayerNorm uses `atol=1e-5, rtol=2e-5` so the same harness
also admits the valid but less accurate FP32 serial accumulation control;
cross-entropy uses `atol=2e-6, rtol=2e-5`. Every element is checked and any
failure remains in the report.

## Exact command

```bash
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions \
  --backends metal --operations layernorm,cross_entropy \
  --metal-subgroup-reductions --pipeline-window 2 \
  --samples 11 --sample-ms 100 --warmup-ms 100 --capture-sources
```

No build, profiler or other GPU workload ran concurrently. The output
directory was new. The driver recaptured/JIT-compiled each shape, checked the
requested policy and plan, downloaded the full output and validated it before
publishing timing.

## Artifact identity

- report `results.json` SHA-256:
  `8d33bc088ba69a62cf72816ad4ec6a22b91ff6577ed65749946a1037f34d939b`
- benchmark executable SHA-256:
  `ab423b2d4a4e7a8683069ea7d5491162ee8718af755a2372737930a0181428f9`
- loaded TIRx bridge SHA-256:
  `4f413da0c70d7b97b465c078b4bb9d42c37c2fbb2a44636a98c3df53cd6ef49c`

The raw report records a dirty worktree because unrelated user edits and the
new evidence directory were present. The measured compiler implementation is
the cited commit; binary, bridge and per-case Metal-source hashes provide the
stronger executable identity.

## Post-measurement source and report validation

The timing artifacts above were not regenerated during documentation work.
The implementation and its report surface were subsequently checked with:

- `ctest --test-dir cmake-build-tirx -R '^test_tile_' --output-on-failure
  -j 1`: **32/32** tests passed in 92.11 seconds;
- the focused CPU guarded-view proof: **1,572 assertions** passed;
- Python benchmark-contract discovery with NumPy installed: **67/67** tests
  passed;
- `xmllint --noout` on `guarded-view-ownership.svg` and the whitespace check
  over handwritten source/documentation changes: both passed; and
- Sphinx 9.1 with `-W --keep-going`: passed after explicitly suppressing only
  the repository's known missing-Doxygen/tutorial warning categories. No
  warning originates in a changed Tile page.

The complete Tile cohort was built and run with the submitted
`metal::mem_flags(3)` source value. The unrelated local `mem_flags(2)` hunk was
restored immediately afterward and remains outside this report snapshot.
Generated `.metal` snapshots deliberately retain the exact trailing bytes
covered by their SHA-256 filenames; they are excluded from the handwritten
whitespace check rather than normalized after measurement.

## Interpretation boundary

This extends the admitted subgroup family from sum/softmax/RMSNorm to forward
LayerNorm and per-row cross-entropy for the listed FP32 contiguous shapes. It
does not cover backward passes, ignored labels, class weights, label
smoothing, arbitrary axes/layouts, low precision, fusion, other Apple GPUs or
pure hardware time. The balanced native reference/candidate causality check
is in the adjacent `m1-max-20260905-metal-subgroup-row-extensions-replay`
directory.
