# XIR/SIMD planner pilot: scalar Tile realization remains far from Torch

The first TileIR→XIR→SIMD planner does **not** close the CPU library gap. Its
automatic map is 38.19× to 54.54× slower than eager Torch by paired median on
this three-shape pilot. Changing root order/block packing alone is insufficient;
the XIR lowerer still expands a 1×1×8 GEMM Tile into scalar SSA without a
packed register-blocked matrix microkernel.

| M×N×K | Planned µs | Fixed µs | Torch µs | Paired planned/fixed [range] | Paired planned/Torch [range] |
|---|---:|---:|---:|---:|---:|
| 32³ | 50.822 | 50.037 | 0.978 | 1.0157× [0.9283, 1.0438] | 51.851× [49.602, 56.017] |
| 128³ | 272.410 | 278.352 | 4.979 | 0.9755× [0.8474, 1.0670] | 54.543× [52.791, 56.695] |
| 127×193×61 | 255.711 | 281.315 | 6.696 | 0.9142× [0.8622, 1.0194] | 38.186× [37.450, 38.751] |

Times are medians of six per-round p50s. Ratios are medians of same-round
ratios, so the ratio of displayed medians may differ. The fixed path is exact
root order `[0,1]`, 64 workers/block. The automatic path searched 12 candidates
(two root-axis permutations × six legal block widths) using the documented
uncalibrated relative-work model. It chose order `[0,1]` for every case and
128 workers at 32³, 1024 workers for the other two. Thus the 127×193×61 change
mostly measures worker packing for one specialization, not discovery of a new
algorithm or memory layout.

## Method and validation

- Compact row-major FP32 C=A×B, alpha=1/beta=0, no transpose or prepacking.
  The same captured TileIR and fixed 1×1×8 specialization are used for both
  XIR paths. Torch uses `torch.mm`; it is a library baseline, not claimed to
  share the same schedule.
- Six rounds cover all six implementation orders of planned/fixed/Torch for
  every shape; shape order rotates. Five samples target 20 ms after 100 ms
  warmup, with eight requested workers/threads. No parameter is selected by
  measured runtime, and no row is discarded.
- Timing is synchronized device-resident host-wall time, including each
  Runtime/API dispatch and synchronization. JIT, allocation/upload and cold
  calls are excluded. This is not a CPU cycle or kernel-only timer.
- All **54/54 outputs passed**: 754,542 checked elements total, `atol=rtol=1e-4`,
  maximum absolute error 0 against an independently recomputed FP64 oracle on
  deterministic dyadic inputs. A second audit reloaded every saved FP32 output,
  regenerated inputs, recomputed the oracle and verified six distinct orders
  per shape.
- `benchmark_tile_xir`, every sibling Luisa dylib/backend module and LLVM
  21.1.8 were SHA-256 fingerprinted before and after; all artifacts were
  unchanged. The benchmark cleared other `LUISA_SIMD_*` controls and fixed
  packet width W8 and eight CPU workers. Torch 2.14.0, NumPy 2.5.2, Apple M1 Max,
  macOS 26.6.2.
- Every XIR row retains the full output, pre-JIT LLVM, command, realization,
  plan/cost breakdown, raw throughput and latency samples, cold/JIT phases and
  SHA-256 values. [results.json](results.json) is authoritative.

Median XIR JIT time in this cohort is 41.166 ms (range 38.397–59.793 ms).
That is reported separately and excluded from warm execution. Normal Tile
compilation now captures pre-JIT LLVM without performing a second assembly
compilation; assembly remains available via the explicit dump control.

## Interpretation

The planner's score is not calibrated in nanoseconds. Its ranking did not
predict a universal measured win over the fixed control: planned was slower
in 4/6 rounds at 32³, 3/6 at 128³ and 1/6 for the ragged case. The ragged-case
median improvement is a useful candidate for independent replay, not evidence
that this cost model generalizes.

The dominant next candidate family is not another root permutation. A useful
CPU GEMM plan needs dependence-safe distribution of output Tile dimensions
across lanes, register blocking, packed/cache-aware reuse and a target matrix
or vector microkernel. The current Schedule reports zero recognized contiguous
buffer reads for these compiled cases, which is a diagnostic to investigate;
it is not a hardware transaction count. Add those legal realizations first,
then calibrate/rank them across GEMM and held-out LLM operator families.

## Reproduction

After a complete CMake build with the SIMD backend and LLVM 21 matching TVM:

```bash
uv run --no-project --python 3.13 --with numpy --with torch \
  python scripts/benchmark/tile_torch/compare_xir.py \
  --native BUILD/bin/benchmark_tile_xir \
  --compiler-artifact /opt/homebrew/opt/llvm@21/lib/libLLVM.dylib \
  --output NEW_EMPTY_DIRECTORY \
  --rounds 6 --samples 5 --sample-ms 20 --warmup-ms 100 --threads 8
```

Use a new output path; the runner refuses to overwrite an existing report.
The exact tested build paths and commands are preserved in `results.json`.

## Packaged technical report update

The self-contained [technical report](report.html) now combines the language,
layout, execution, memory, pipeline, TileIR/runtime boundary, XIR pilot, MPP
cost-model study and latest frozen Metal replay. Its canonical
[artifact input](artifact.json) has 20 ordered report sections, eight metric
cards, three native charts, four audit tables, nine bounded datasets and nine
source records. The full design remains in the repository documents; this
artifact is the answer-first implementation/evidence view rather than a second
independent specification.

The revision preserves the failed MPP cost v1 result. The new comparison chart
shows shape-wise finite-set regret for v1 and v2; the adjacent exact table keeps
both model and measured schedule choices. It labels the data as three-sample,
10 ms in-cohort calibration. The Metal chart and eight-shape table use the
independent `m1-max-20260905-mpp-cost-v2-replay` evidence: 784/784 validated
outputs, MPP views faster than Torch and MPS on 8/8 shapes, and 270.675 µs at
1024³ versus 272.572 µs MPS and 284.654 µs Torch.

Chart map and visual QA:

- **XIR gap:** three paired planned/Torch ratios show why root permutation and
  worker packing do not replace a packed CPU realization.
- **MPP v1 versus v2 regret:** grouped bars answer whether the subgroup
  critical-path/wave correction improves ranking. The visible caveat prevents
  interpreting its calibration cohort as held-out evidence.
- **1024³ Metal paths:** seven synchronized host-wall medians keep compiler and
  runtime ownership separate. Exact eight-shape schedules and paired ratios
  remain in the adjacent table because their absolute times span two orders of
  magnitude.

The canonical portable-report validator and packager passed. Its verification
stage is `structural_only` because no compatible Chromium headless-shell is
installed; exact payload equality, reader/runtime roots and semantic chart/table
fallbacks passed, but desktop/narrow viewport and source-dialog interaction were
not browser-tested. No browser was downloaded. Final SHA-256 values are
`7b6f3108e3b217e0ac5d3db9fa753d444b370495320b06da6808534d8fca2d40`
for `artifact.json` and
`c169c6d26a15bd36c3d285791a8588305ea4e5b0e9f6d7f0adc53522684075e5`
for `report.html`.
