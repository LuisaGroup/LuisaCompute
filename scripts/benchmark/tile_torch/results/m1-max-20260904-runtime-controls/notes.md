# Same-source TIRx Runtime controls

## Outcome

The same Metal backend factory now supports explicit native MPP and TIRx
routes, returning ordinary Luisa Runtime shaders. The standalone TVM-runtime
TIRx path remains supported and tested. The bridge exports typed argument
provenance and static launch extents; it does not parse or rewrite Metal source.

This is infrastructure and diagnosis progress, **not** a large-GEMM speedup.
For 1024³ the synchronized host-wall throughput medians are:

| Path | µs/call |
|---|---:|
| TileIR → native MPP → Luisa Runtime | 294.958 |
| TileIR → TIRx → TVM runtime | 320.225 |
| TileIR → TIRx → Luisa Runtime, fast math off | 320.784 |
| TileIR → TIRx → Luisa Runtime, fast math on | 320.667 |
| Handwritten MPP | 272.627 |
| Direct MPS | 279.207 |
| Eager Torch MPS | 291.332 |

Paired-round TIRx/Luisa-to-TIRx/TVM time ratios are 1.0026 (fast math off)
and 1.0013 (on). Changing these downstream paths did not close the large
matrix gap in this experiment. That supports prioritizing the device-side
matrix realization/schedule over treating TVM runtime overhead as the main
cause. It does not prove which GPU instruction/resource is the bottleneck.
At 32³ and 128³ the Luisa path has measurable additional host-wall overhead;
this integration is not uniformly faster. Native MPP also remains about
5.6% slower than direct MPS on 1024³ by paired median time ratio.

## Protocol and scope

- 8 shapes × 14 rounds × 7 implementations = **784/784 full outputs valid**.
- All 112 case/round groups have identical generated-source SHA-256 across
  the three TIRx routes. Eight unique Metal sources are saved in `sources/`.
- Exact realized threadgroup widths match the independent TVM planner report.
  The Runtime exporter obtains its grid/block from typed static launch data;
  the comparison runner checks threadgroup width, source identity and full
  output, not a separately instrumented TVM dispatch-grid trace.
- Seven rotations and their reversals balance every position and pairwise
  precedence over 14 rounds. Configurations are frozen; there is no search
  or fastest-round selection. Raw outliers are retained.
- Seven throughput samples per path, 30 ms target duration, 200 ms warmup;
  synchronized host wall includes dispatch/encoding/submission/synchronization,
  and excludes compilation/allocation/transfers. No GPU-only claims or mixed
  host-wall/GPU-interval speedups are made.
- Native and handwritten MPP share fixed atom/cohort configurations. TIRx uses
  the independently frozen joint-search schedule. The two TIRx Runtime variants
  and TVM path use the same capture routine and identical generated source.
- Native/handwritten MPP disable fast math and relaxed precision. TIRx/Luisa
  explicitly tests both fast-math settings. The local TVMx Metal runtime source
  (`src/backend/metal/runtime/metal_module.mm`) requests fast math on; its policy
  is not overridden by this runner. MSL language/resource compiler settings and
  runtime submission mechanisms can still differ. Equal source is not a claim
  of equal compiled binaries.
- Inputs use the existing dyadic pattern and full NumPy FP64 GEMM oracle with
  `atol=rtol=1e-4`. These inputs alone cannot certify input multiplication
  precision. Separate Runtime tests use non-dyadic sine/cosine data, changed
  inputs, transposes, nonzero byte offsets, tails and guard regions.
- The recorded benchmark binaries and their adjacent shared libraries remained
  unchanged throughout the run. External TVMx/Torch/system framework versions
  remain dependencies; the adjacent-library manifest is not a hermetic bundle
  of every transitive dependency. No builds or GPU tests ran concurrently.

Command:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/compare_lowerings.py \
  --native cmake-build-tirx/bin/benchmark_tile_native \
  --tirx cmake-build-tirx/bin/benchmark_tile_tirx \
  --mpp cmake-build-tirx/bin/benchmark_tile_mpp \
  --mps cmake-build-tirx/bin/benchmark_tile_system \
  --mpp-plan scripts/benchmark/tile_torch/results/m1-max-20260904-mpp-search/results.json \
  --tirx-plan scripts/benchmark/tile_torch/results/m1-max-20260904-joint-search/results.json \
  --tirx-runtime-controls --rounds 14 --samples 7 --sample-ms 30 --warmup-ms 200 \
  --output scripts/benchmark/tile_torch/results/m1-max-20260904-runtime-controls
```

## Regression checks

The full selected CMake build completed before tests/timing. The selected
21 Tile/native/TIRx CPU/Metal CTests passed with both Luisa and Metal validation
enabled; the benchmark Python suite passed 41 tests. The ordinary SIMT DSL
sugar executable also passed on Metal. No C++ implementation changes occurred
during or after the measured run.

Two pre-existing source assertions in the Metal cooperative/memory suites
still conflict with the unowned `mem_flags(3)` → `mem_flags(2)` working-tree
change and were excluded from this selected run; neither the hunk nor the
assertions was modified. This is **not** a claim that the complete suite is
green. The earlier private-prefetch WIP remains unchanged; frozen schedules
use pipeline window one, so this campaign does not claim a prefetch benefit.

The TIRx Runtime adapter currently rejects multiple/conditional host launches,
host effects/allocations, dynamic launch resources, non-FP32 arguments and
AOT/indirect execution. The wider standalone bridge is retained. A general
Machine TileIR and a calibrated common native/TIRx cost model remain follow-up
work, not completed by this device-artifact adapter.
