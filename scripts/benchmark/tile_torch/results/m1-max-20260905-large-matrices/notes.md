# Scale coverage exposes a remaining GEMM gap

## Conclusion

The old 1024-cubed result does not generalize into a library-performance claim.
At **8192³**, frozen native Tile→MPP takes **1.985× Torch's GPU-control time**
by median same-round ratio; TIRx→MPP with proved views reduces that to
**1.125×**, but still loses all 14 GPU pairs. The view request also rejects
two large tail shapes. All seven paths and every rejection remain in the
evidence, including valid other paths on those two shapes.

Wide softmax, RMSNorm and affine LayerNorm pass all 72 complete-output checks
and beat eager Torch in both rounds of GPU and batched E2E throughput for
all 18 cases. This is **two-round scale coverage**, not stable all-shape
acceptance. Large RMSNorm margins are small, and single-call latency retains
mixed results and one consistently slower case. The general goal stays open.

The [predeclared protocol](protocol.md) freezes workloads and configurations;
[environment preflight](environment-preflight.md) separates missing compiler
capability from actual schedule admission. No new measurements selected a
schedule or fitted a cost coefficient. No C++ implementation was rebuilt or
changed for this cohort.

## Measurement contract and evidence

Recorded September 5–6, 2026 (local time), Apple M1 Max, 64 GiB unified memory,
macOS 26.6.2, FP32 compact row-major inputs. GEMM is C=A*B, alpha=1, beta=0,
no transpose, prepacking or reduced precision. Native/handwritten MPP disable
fast/relaxed math; TVM Metal's existing fast-math behavior is unchanged. Torch
is eager 2.14.0, commit `08187d9e0fba026dc8217405802ab5381dc88d90`, Python 3.13.7;
eight host threads are requested, not measured library worker counts.

Four distinct metrics are retained:

- **GPU batch:** no-counter command-buffer GPU intervals per invocation.
- **E2E batch:** uninstrumented host-wall time per invocation amortized over
  dispatches and synchronization.
- **GPU single:** no-counter command-buffer GPU intervals for one invocation.
- **E2E single:** individually synchronized host-wall dispatch-to-completion.

GPU control includes GPU work and gaps within command buffers; it is **not
an isolated kernel timestamp**. Handwritten MPP uses its direct timestamp
implementation; other paths use the shared Metal helper's no-counter phase.
Instrumented compute-pass probes are diagnostic only. Setup allocations,
uploads, JIT and the FP64 oracle are outside warm timing. Native output is
preallocated. Torch GEMM/softmax use `out=`; functional norms allocate returned
output during timing. Fusion/API differences remain part of that comparison.

Absolute values below are medians of within-round p50s; paired ratios are
medians of same-round time ratios, **not ratios of displayed medians**.
Ranges are observed extrema, not confidence intervals. GPU and host phases
are separate measurements: subtracting their medians does not measure dispatch
overhead. Large GEMM times drift substantially even with balanced order. The
cause was not measured; these data do not establish a thermal or occupancy
diagnosis, nor justify fitting percent-level model corrections.

| Evidence | Coverage | Validation / failure boundary |
|---|---|---|
| [GEMM replay](gemm-replay/results.json) | 6 shapes × 7 paths × 14 rounds | 560 complete outputs pass; 28 fixed-view admission rejections retained |
| [Corrected view pilot](view-pilot-patched/results.json) | 6 fixed requests | 8 complete outputs pass; 2 admission rejections; timing is exploratory |
| [Initial capability probe](view-pilot/results.json) | 6 requests | All lack the required TVM MPP contract; not numerical checks |
| [Forward reductions](reduction-forward/results.json) and [reverse reductions](reduction-reverse/results.json) | 3 operators × 6 shapes × 2 paths × 2 rounds | All 72 complete outputs pass |
| [Independent audit](audit.py) and [receipt](audit.json) | Order, arithmetic, plans, source hashes, artifact hashes, all four metrics | Evidence is consistent **with reported rejections**, not all-paths-pass |

The independent audit recomputes timing summaries from raw samples without
importing benchmark statistics helpers. It verifies recorded full-output
validation, not saved output arrays: transient arrays were not retained.
GEMM checks 11,022,491,732 output elements across the replay. Its dyadic input
family happens to yield zero maximum absolute error on all validated outputs;
this does not prove general-distribution FP32 accuracy. Maximum native errors
are 1.31e-10 for softmax, 1.87e-7 for RMSNorm and 2.62e-7 for LayerNorm; Torch's
corresponding maxima are 9.74e-11, 1.95e-7 and 3.16e-7. Tolerances are unchanged.

## GEMM: frozen schedules at larger dimensions

Path names in the tables:

- **Native:** native Tile→MPP; **Hand:** independent handwritten MPP. Both
  retain configuration `32,32,1,1,0,1,4,4` from the earlier 1024³ control.
- **TIRx:** ordinary SIMD-group matrix lowering, 32×32×32, 128 threads.
- **MPP:** TIRx→MPP without forwarded inputs, same 32×32×32/128-thread request.
- **Views:** TIRx→MPP with proved input views, old 1024³ winner unchanged:
  128×32×1024, 128 threads. Both TIRx variants use window 1 and copy batch 1.
- **MPS:** direct MPSMatrixMultiplication, `MPSKernelOptionsNone`;
  **Torch:** eager MPS `mm(..., out=...)`.

These are fixed-schedule scale tests, not each path's newly tuned optimum.
Every implementation occupies each of seven positions twice per shape, with
7:7 pair precedence and rotating shape order. Each round uses five samples,
20 ms windows and 100 ms warmup. Full output validation precedes acceptance.

### GPU batch time (milliseconds, lower is better)

| M×N×K | Native | TIRx | Hand | MPS | Torch | MPP | Views |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×2048×2048 | 3.201 | 4.089 | 2.832 | 3.018 | 2.989 | 3.713 | 3.012 |
| 4096×4096×4096 | 29.978 | 39.556 | 27.169 | 30.148 | 27.074 | 33.065 | 29.273 |
| 8192×8192×8192 | 476.117 | 438.198 | 412.663 | 248.050 | 237.612 | 421.804 | 271.092 |
| 256×11008×4096 | 6.581 | 6.128 | 6.902 | 3.944 | 3.810 | 5.718 | 4.264 |
| 4096×4096×11008 | 102.115 | 156.372 | 90.646 | 94.582 | 82.550 | 150.774 | rejected |
| 2049×4097×1025 | 3.626 | 10.898 | 3.748 | 3.645 | 3.350 | 10.581 | rejected |

### E2E batch time (milliseconds, lower is better)

| M×N×K | Native | TIRx | Hand | MPS | Torch | MPP | Views |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×2048×2048 | 3.254 | 4.208 | 2.883 | 3.012 | 3.196 | 3.596 | 3.144 |
| 4096×4096×4096 | 30.026 | 35.439 | 27.701 | 30.651 | 27.377 | 32.446 | 29.808 |
| 8192×8192×8192 | 461.686 | 369.682 | 412.923 | 295.217 | 278.915 | 399.020 | 300.684 |
| 256×11008×4096 | 6.529 | 6.214 | 7.156 | 4.101 | 3.896 | 6.005 | 4.276 |
| 4096×4096×11008 | 95.074 | 146.511 | 91.488 | 85.497 | 77.179 | 131.272 | rejected |
| 2049×4097×1025 | 3.984 | 11.906 | 3.865 | 3.811 | 3.520 | 11.755 | rejected |

### Paired view-path comparison

Ratio <1 favors Views. Counts are faster rounds out of 14. Other paths'
paired comparisons against both MPS and Torch, every round and all four
metrics are retained in `audit.json`; no valid non-view result is discarded
because another path failed.

| M×N×K | GPU Views/MPS [range]; faster | GPU Views/Torch [range]; faster | E2E Views/Torch [range]; faster |
|---|---|---|---|
| 2048×2048×2048 | 0.991 [0.712, 1.490]; 8/14 | 1.098 [0.644, 1.408]; 6/14 | 1.032 [0.687, 1.486]; 7/14 |
| 4096×4096×4096 | 1.044 [0.776, 1.292]; 6/14 | 1.052 [0.722, 1.521]; 6/14 | 1.057 [0.691, 1.640]; 5/14 |
| 8192×8192×8192 | 1.063 [0.904, 1.220]; 5/14 | 1.125 [1.019, 1.251]; 0/14 | 1.096 [0.887, 1.597]; 2/14 |
| 256×11008×4096 | 1.058 [0.560, 1.377]; 5/14 | 1.170 [0.900, 1.414]; 4/14 | 1.113 [0.867, 1.426]; 4/14 |
| 4096×4096×11008 | rejected | rejected | rejected |
| 2049×4097×1025 | rejected | rejected | rejected |

Native's 8192³ GPU/Torch ratio is 1.985 [1.916, 2.786], slower in 14/14;
its E2E/Torch ratio is 1.680 [1.548, 2.919], also slower in 14/14. At that
shape ordinary TIRx, non-forwarding TIRx→MPP and handwritten MPP all lose
every GPU and E2E pair against both external baselines. This robust gap is
more actionable than the noisy near-parity 2048³/4096³ medians.

### Single-call timings (milliseconds)

GPU single:

| M×N×K | Native | TIRx | Hand | MPS | Torch | MPP | Views |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×2048×2048 | 2.929 | 4.015 | 2.876 | 2.953 | 2.712 | 3.493 | 3.168 |
| 4096×4096×4096 | 30.806 | 44.445 | 27.346 | 29.699 | 27.329 | 36.394 | 29.557 |
| 8192×8192×8192 | 482.758 | 435.066 | 445.393 | 266.238 | 248.859 | 417.100 | 288.433 |
| 256×11008×4096 | 6.628 | 6.202 | 7.055 | 3.911 | 3.665 | 5.777 | 4.327 |
| 4096×4096×11008 | 109.504 | 133.046 | 92.606 | 91.661 | 85.853 | 133.052 | rejected |
| 2049×4097×1025 | 3.586 | 10.367 | 3.621 | 3.369 | 3.369 | 10.159 | rejected |

E2E single:

| M×N×K | Native | TIRx | Hand | MPS | Torch | MPP | Views |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×2048×2048 | 3.599 | 4.970 | 3.257 | 3.578 | 3.545 | 4.063 | 3.771 |
| 4096×4096×4096 | 30.476 | 36.739 | 27.930 | 31.258 | 28.383 | 32.681 | 30.746 |
| 8192×8192×8192 | 498.179 | 391.083 | 445.749 | 299.768 | 291.875 | 376.265 | 321.949 |
| 256×11008×4096 | 7.239 | 6.764 | 7.489 | 4.593 | 4.316 | 6.431 | 4.890 |
| 4096×4096×11008 | 99.076 | 173.238 | 92.974 | 88.598 | 79.112 | 157.707 | rejected |
| 2049×4097×1025 | 4.533 | 11.748 | 4.341 | 4.371 | 3.815 | 11.777 | rejected |

### Why the fixed view request rejects tails

The diagnostic is `no legal Metal MPP group plan`. Both rejected cases have
K tails relative to BK=1024. In `views.cpp`, forwarding requires proving the
source region fully in bounds when MPP disables preserved view guards in
`compiler.cpp`. A tail therefore cannot retain that unguarded address view.
The fixed 128×32×1024 logical A/B tiles would total 640 KiB if both had to be
materialized as FP32, while `planner.cpp` admits only candidates satisfying
local matrix geometry, fragment cost and target shared-memory limits.

This code-path explanation concerns the **fixed large-view schedule**, not
MPP support in general: native/handwritten MPP and the smaller non-forwarding
32×32×32 TIRx request validate both shapes. No replacement block is silently
selected for the view column. All 28 replay rejections are therefore expected
coverage failures; the replay exits 1, not an all-green result.

## Reductions: wide rows and large working sets

Six new shapes per operator, one forward and one reverse case order, reversed
native/Torch precedence per case. Five samples, 30 ms windows, 100 ms warmup.
The automatic analytic family is unchanged: V4, ordered U1, preserved shared
Tile SSA, opt-in immutable input caching, and a 64-scalar private storage
bound. No fixed S2/P4 mapping is extrapolated from earlier short-row results.

The audit independently enumerates all 39 automatic candidates and verifies
the exact minimum-score legal choice and source identity in both rounds.
S/P means SIMD groups per program / programs per threadgroup. Every chosen
P is 1. Widths 8191/8192 choose S16 (512 workers); width 16384 chooses S26
(832 workers); width 4096 chooses S11 for softmax/LayerNorm and S16 for RMSNorm.
Audited stripe storage remains 8–40 scalars, not one full-width array per worker.
These are planner resource estimates, not measured register allocation.

The actual generated MSL is also checked: softmax has two `simd_max` and two
`simd_sum` call sites; RMSNorm has two `simd_sum`; LayerNorm has four.
There is one threadgroup barrier per logical reduction (one for RMSNorm,
two for softmax/LayerNorm). These are cooperating two-level reductions,
not a serial row per worker. Source calls do not establish final ISA.

GPU batch absolute times are µs; ratios are native/Torch, lower is better.
All 18 GPU and E2E batch cases favor native in both observed rounds.

| Operator / rows×width | S/P | GPU µs | Torch GPU µs | GPU ratio [range] | E2E µs | E2E ratio [range] |
|---|---|---:|---:|---|---:|---|
| softmax_37x8191 | 16/1 | 11.438 | 27.084 | 0.423 [0.404, 0.441] | 12.370 | 0.356 [0.351, 0.361] |
| softmax_1024x8192 | 16/1 | 154.548 | 254.706 | 0.607 [0.594, 0.619] | 158.275 | 0.583 [0.579, 0.588] |
| softmax_1024x16384 | 26/1 | 395.407 | 573.559 | 0.689 [0.685, 0.694] | 402.646 | 0.669 [0.662, 0.676] |
| softmax_4096x8192 | 16/1 | 792.019 | 1006.430 | 0.787 [0.786, 0.788] | 817.065 | 0.768 [0.764, 0.773] |
| softmax_8192x4096 | 11/1 | 789.845 | 889.418 | 0.888 [0.887, 0.889] | 807.288 | 0.854 [0.850, 0.858] |
| softmax_16384x4096 | 11/1 | 1579.533 | 1746.604 | 0.904 [0.904, 0.904] | 1616.126 | 0.878 [0.877, 0.880] |
| rmsnorm_37x8191 | 16/1 | 11.629 | 13.288 | 0.875 [0.872, 0.878] | 12.137 | 0.707 [0.703, 0.710] |
| rmsnorm_1024x8192 | 16/1 | 155.291 | 162.342 | 0.957 [0.949, 0.964] | 159.454 | 0.941 [0.940, 0.942] |
| rmsnorm_1024x16384 | 26/1 | 405.528 | 414.320 | 0.979 [0.978, 0.979] | 411.457 | 0.967 [0.962, 0.973] |
| rmsnorm_4096x8192 | 16/1 | 795.076 | 807.782 | 0.984 [0.976, 0.993] | 813.432 | 0.984 [0.979, 0.989] |
| rmsnorm_8192x4096 | 16/1 | 790.359 | 800.068 | 0.988 [0.987, 0.989] | 812.588 | 0.983 [0.979, 0.987] |
| rmsnorm_16384x4096 | 16/1 | 1594.521 | 1612.320 | 0.989 [0.984, 0.994] | 1614.056 | 0.972 [0.969, 0.974] |
| layernorm_37x8191 | 16/1 | 12.652 | 34.603 | 0.366 [0.362, 0.370] | 13.882 | 0.335 [0.335, 0.336] |
| layernorm_1024x8192 | 16/1 | 160.715 | 507.247 | 0.317 [0.316, 0.318] | 164.832 | 0.314 [0.314, 0.314] |
| layernorm_1024x16384 | 26/1 | 420.823 | 1216.406 | 0.349 [0.315, 0.383] | 426.470 | 0.350 [0.315, 0.384] |
| layernorm_4096x8192 | 16/1 | 803.643 | 2038.785 | 0.394 [0.393, 0.395] | 837.119 | 0.398 [0.392, 0.405] |
| layernorm_8192x4096 | 11/1 | 797.663 | 1597.348 | 0.499 [0.497, 0.502] | 813.556 | 0.498 [0.495, 0.500] |
| layernorm_16384x4096 | 11/1 | 1587.361 | 3164.560 | 0.502 [0.499, 0.504] | 1628.853 | 0.500 [0.489, 0.511] |

The automatic family remains feasible at 16384-wide rows and a 512 MiB
input/output payload, but the RMSNorm GPU advantage shrinks to about 1–4%
on the five larger cases. That is not a robust all-device efficiency margin.
Nor is throughput latency: GPU single-call has 16 cases faster in both rounds
and two mixed RMSNorm cases. E2E single-call has 14 faster in both, three mixed,
and RMSNorm 37×8191 slower in both (1.065 [1.034, 1.097]). RMSNorm
1024×16384 E2E latency is especially variable, 1.169 [0.883, 1.454]. All
single-call samples and ratios remain in the audit receipt.

Torch probe/control command-buffer ratios range 0.845–1.236 for GEMM and
0.804–2.863 for reductions across throughput/single-call phases. These are
observed inter-phase ratios, not causal instrumentation-overhead estimates.
No compute-pass probe replaces the no-counter values in these tables.

## Reproduction, artifact boundary and next work

The GEMM replay records 26 executable/library/helper artifacts before and after
timing, all unchanged. It loads the five patched TVM/FFI libraries explicitly
from `/tmp/luisa-tvm-mpp.VaKmzx/build/lib`. Reductions retain the ordinary wheel
environment and their own 22-artifact before/after snapshots, also unchanged.
The current-artifact audit confirms both sets. These are an explicit inventory,
not a claim to fingerprint the operating system or every Torch component.
The reused TIRx executable SHA256 is
`9cbdc7873355118a9874c58eec499cdb0692dd9286c14b52afceae620c62ad87`.
The dirty source checkout is not relabeled as a freshly built executable.

Commands are run from the repository root, one GPU workload at a time. The
protocol and environment note precede timings; exact path order, settings,
generated sources and raw samples are in the linked JSON. GEMM invocation:

```sh
env DYLD_LIBRARY_PATH=/tmp/luisa-tvm-mpp.VaKmzx/build/lib \
uv run --no-project --python 3.13 --with numpy --with torch python \
  scripts/benchmark/tile_torch/compare_lowerings.py \
  --native cmake-build-tirx/bin/benchmark_tile_native \
  --tirx cmake-build-tirx/bin/benchmark_tile_tirx \
  --mpp cmake-build-tirx/bin/benchmark_tile_mpp \
  --mps cmake-build-tirx/bin/benchmark_tile_system \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-large-matrices/gemm-replay \
  --shape 2048x2048x2048 --shape 4096x4096x4096 --shape 8192x8192x8192 \
  --shape 256x11008x4096 --shape 4096x4096x11008 --shape 2049x4097x1025 \
  --mpp-config 32,32,1,1,0,1,4,4 --tirx-mpp --tirx-view-block 128,32,1024 \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_compiler.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime_metal.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime_extra.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_ffi.dylib \
  --compiler-artifact /opt/homebrew/opt/llvm@21/lib/libLLVM.dylib \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --rounds 14 --samples 5 --sample-ms 20 --warmup-ms 100 --timeout 300 --threads 8
```

Forward reductions use the command below. Reverse uses `reduction-reverse`
and row order `16384x4096,8192x4096,4096x8192,1024x16384,1024x8192,37x8191`;
all other options remain identical. Do not apply the GEMM loader override.

```sh
uv run --no-project --python 3.13 --with numpy --with torch python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-large-matrices/reduction-forward \
  --backends metal --operations softmax,rmsnorm,layernorm \
  --row-shapes 37x8191,1024x8192,1024x16384,4096x8192,8192x4096,16384x4096 \
  --metal-subgroup-reductions --input-views --cache-reduction-inputs \
  --reduction-lane-elements 4 --reduction-unroll 1 \
  --group-threads 0 --reduction-programs-per-group 0 \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --samples 5 --sample-ms 30 --warmup-ms 100 --timeout 300 --capture-sources

uv run --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260905-large-matrices/audit.py \
  --check-current-artifacts
```

Use fresh output directories for new timings; do not overwrite this cohort.
Without `--check-current-artifacts`, the audit checks committed evidence and
recorded before/after identities without requiring original local binaries.
The receipt is exclusive-create and is not rewritten by a later audit.

The benchmark orchestration passes 93 Python tests after adding explicit
GEMM shape selection and GPU-control extraction tests. This is not a new
build/CTest claim: the earlier 31/33 CTest boundary and two unrelated
barrier-source assertions remain unchanged.

Next work should separate two questions: (1) admitting a bounded full/tail
memory realization for the large-view request, and (2) independently searching
matrix reuse, grouping and resource choices that scale beyond the old 1024³
schedule. Native MPP is a separate realization and does not automatically
inherit the TIRx planner. Wide-row RMSNorm needs a longer latency/throughput
replay before tiny margins influence defaults. None of these fixes is claimed
implemented by this measurement-only experiment.

Report QA contract: the existing Sphinx performance section owns the reader
narrative; this note owns cohort methods and provenance. Exact neutral tables
show all paths and failures, with GPU/E2E and throughput/latency separate.
The headline is the remaining gap, not a selected winning shape. Source hashes,
order balance, emitted collectives, arithmetic and numerical records are audited
independently; desktop/mobile documentation rendering is checked separately.

Final QA: all 78 new note/Sphinx table rows match the independent audit's
rounded values and retained rejections. Five in-memory negative probes
reject a missing round, unbalanced order, a hidden rejection, nonfinite time
and an instrumented control. The 93-test Python discovery run passes. A fresh
Sphinx build retains only the ten existing missing-Doxygen-XML warnings;
48 generated pages, 3,609 local links/assets and 199 compatibility anchors
pass the documentation checker. Desktop 1440×1050 and mobile 390×844 renders
are inspected: the new compact tables show all columns on desktop and allow
scrolling to the final column on mobile, without page overflow. The existing
site hierarchy and theme are retained.
