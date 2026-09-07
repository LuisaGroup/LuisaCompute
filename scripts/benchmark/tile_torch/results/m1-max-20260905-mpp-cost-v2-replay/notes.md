# Frozen Metal MPP cost-model v2 replay

Date: 2026-09-05 Asia/Shanghai (`2026-09-04T20:00:52.724465Z` in the raw report).
Machine: Apple M1 Max, macOS 26.6.2 arm64. Source revision:
`b8c3c54f81f2a4ad947e295f1f75e57605bf8833` plus recorded uncommitted changes.

## Result

The separately searched MPP-v2 view schedules were frozen, recompiled and run
in 14 position-balanced rounds against six independent controls. All 784
complete outputs (8 shapes × 14 rounds × 7 paths) passed the same FP64 oracle.
All 21 fingerprinted binaries/compiler/runtime artifacts were unchanged across
the run.

TIRx→MPP with proved read-only view forwarding beats eager Torch and direct
MPS on all eight tested FP32 GEMMs by paired-round median. At 1024³ it measures
270.675 µs/call, 0.62% faster than MPS (paired ratio 0.9938×) and 4.87% faster
than Torch (0.9513×); it is still 1.68% slower than handwritten MPP. This closes
the previous frozen 64×64 plan's small 1024³ MPS gap for this cohort. It does
not establish cross-device, low-precision or non-GEMM library parity.

## Balanced throughput and frozen schedules

Times are medians of the 14 per-round throughput medians. Ratios are medians of
same-round ratios; they are not ratios of the displayed medians or confidence
intervals. `BM×BN×BK @ threads` is the frozen TIRx-view schedule.

| M×N×K | Frozen TIRx-view schedule | TIRx views µs | Hand MPP µs | MPS µs | Torch µs | Paired view/MPS | Paired view/Torch |
|---|---|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 32×32×32 @ 128t | 2.982 | 2.809 | 10.081 | 26.899 | 0.2794× | 0.1105× |
| 128×128×128 | 32×32×128 @ 256t | 5.335 | 5.441 | 16.904 | 27.218 | 0.3174× | 0.1943× |
| 512×512×512 | 32×64×32 @ 128t | 42.413 | 46.802 | 52.428 | 47.745 | 0.8285× | 0.8919× |
| 1024×1024×1024 | 128×32×1024 @ 128t | 270.675 | 266.105 | 272.572 | 284.654 | 0.9938× | 0.9513× |
| 256×1024×128 | 64×64×128 @ 256t | 16.025 | 17.286 | 20.350 | 28.668 | 0.8189× | 0.5554× |
| 1024×128×256 | 32×32×32 @ 128t | 16.500 | 18.508 | 26.270 | 28.655 | 0.5946× | 0.5596× |
| 127×193×61 | 32×32×32 @ 256t | 8.861 | 7.127 | 16.915 | 26.997 | 0.5172× | 0.3266× |
| 513×257×129 | 32×32×32 @ 256t | 20.607 | 24.424 | 35.043 | 34.002 | 0.5874× | 0.6057× |

At 1024³ the TIRx-view path was slower than MPS in only 1 of 14 rounds. The
ragged 127×193×61 case is a useful warning: the frozen view schedule beats both
external libraries but is slower than the handwritten and native-MPP controls.
The report therefore retains all seven paths rather than presenting one blended
“Metal” number.

## Paths and attribution

1. `tile_native_mpp`: Candidate TileIR lowered by the native Metal backend.
2. `tile_tirx`: ordinary TIRx/TVM SIMD-group lowering.
3. `handwritten_mpp`: direct MPP tensor-operations control.
4. `mps`: direct `MPSMatrixMultiplication` control.
5. `torch`: eager PyTorch MPS.
6. `tile_tirx_mpp`: TIRx's patched MPP codegen with the old frozen TIRx schedule.
7. `tile_tirx_mpp_views`: the separately frozen MPP-v2 schedule with proved
   immutable input views.

The selected path reports `cost_basis=metal_mpp_memory_v2`, actual MPP call
sites and zero SIMD-group MMA calls. This score is a relative analytic prior;
the table is measured synchronized host-wall time. Native/hand MPP use fast
math off; TVM's current Metal runtime hardcodes fast math on. The comparison is
therefore a system-level performance target, not an isolated arithmetic-only
kernel-time claim.

## Exact replay command

```bash
uv run --no-project --python 3.13 --with numpy --with torch python \
  scripts/benchmark/tile_torch/compare_lowerings.py \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_native \
  --tirx /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_tirx \
  --mpp /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_mpp \
  --mps /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_system \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-replay \
  --mpp-plan scripts/benchmark/tile_torch/results/m1-max-20260904-mpp-search/results.json \
  --tirx-plan scripts/benchmark/tile_torch/results/m1-max-20260904-joint-search/results.json \
  --tirx-mpp \
  --tirx-view-plan scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-search/results.json \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_compiler.dylib \
  --rounds 14 --samples 7 --sample-ms 30 --warmup-ms 200 --threads 8
```

The three frozen plan SHA-256 values and all artifact hashes are in
`results.json`. No schedule search, failed-row removal, minimum-of-rounds
selection, build, test or profiler ran during this replay.

## Post-replay regression checks

After timing completed, the incremental CMake build passed and the complete
`ctest -R '^test_tile_'` cohort passed 32/34 tests. The only failures were the
already known `test_tile_tirx_cooperative_metal` and
`test_tile_tirx_memory_metal` memory-flag expectations; all planner, matrix,
native Runtime, XIR Runtime, LLM and CPU/Metal PoC tests passed. The Python
benchmark-orchestration suite passed 61/61 tests. A strict Sphinx build with
warnings as errors also passed after the new evidence links and diagram were
added. These checks ran after—not concurrently with—the performance replay.

## What remains open

- The v2 search evidence is in-cohort; held-out shape and operator regret is
  still required before treating its coefficients as a reusable device profile.
- Residual model regret reaches 34.37% inside the search cohort, so analytic
  ranking remains a shortlist prior rather than the final authority.
- CPU remains structurally behind Torch/Accelerate and needs a packed,
  cache-aware vector/matrix realization—not coefficient polishing alone.
- Production LLM shapes, FP16/BF16, fused epilogues, reductions, softmax and
  attention still need equivalent XIR/TIRx/Torch/library comparisons.
