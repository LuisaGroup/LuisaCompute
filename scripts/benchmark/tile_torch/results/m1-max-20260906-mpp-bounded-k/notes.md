# Bounded-K MPP views: proof, compatibility and fixed-plan comparison

This checkpoint adds a legal realization, not a new DSL primitive or a fitted
cost model. For a proved common zero-padded K suffix, TIRx now constructs
bounded immutable A/B inline tensors. It can admit the existing 128×32×1024
view schedule without its nominal 640 KiB A/B shared staging. M/N must still
be fully in bounds. The general MPS/Torch performance goal remains open.

See the [predeclared protocol](protocol.md), [raw seven-path replay](replay/results.json),
[independent audit](audit.json), [audit program](audit.py), and
[frozen-baseline admission record](admission.json). The separate
[MPP scope screen](../m1-max-20260906-mpp-scope/notes.md) retains a negative
result: larger collective participation is not a universal improvement.

## Results: admission improves, library parity remains open

All **392 complete outputs** pass (8,325,201,920 checked output elements;
maximum absolute error 0 for these dyadic inputs). All **26** executable,
adjacent-library, timing-helper and external TVM/FFI artifact hashes remain
unchanged. The independent audit verifies all four metrics, balanced order,
fixed plans and archived sources, and rejects eight intentionally corrupted
records: missing row, wrong order, invalid output, NaN time, missing source,
wrong element count, instrumented control and changed artifacts.

The view plan has zero shared allocation for all four shapes. Against the
separate materialized 32×32×32 MPP control, paired GPU time ratios are
0.677/0.439/0.739/0.776, with every GPU pair improving. This is a **different-
geometry comparison**, not the isolated effect of the new proof. The three
nontrivial shapes still have paired GPU ratios above one against both
MPS and Torch. The small shape wins every host-throughput pair but loses
five of fourteen GPU-throughput pairs to MPS.

### Seven-route GPU batch time

Command-buffer microseconds, lower is better; medians of 14 round p50s.
“MPP” is TIRx with materialized inputs; “Views” is the separately fixed
128×32×1024 TIRx MPP schedule. Native/Hand use the same fixed native geometry.

| M×N×K | Native µs | TIRx µs | Hand µs | MPS µs | Torch µs | MPP µs | Views µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128×128×61 | 9.148 | 12.933 | 7.968 | 8.938 | 13.889 | 12.603 | 8.592 |
| 1024×1024×1537 | 500.481 | 1256.384 | 482.893 | 433.547 | 437.150 | 1164.593 | 511.423 |
| 4096×4096×11008 | 73837.125 | 90557.271 | 72699.500 | 53208.125 | 54887.771 | 87732.792 | 60221.083 |
| 8192³ | 412910.000 | 271885.438 | 378928.313 | 220077.896 | 210055.750 | 291378.813 | 241151.958 |

| M×N×K | GPU View/MPS [range] | Slower / 14 | GPU View/Torch [range] | Slower / 14 |
|---|---:|---:|---:|---:|
| 128×128×61 | 0.962 [0.911, 1.089] | 5 | 0.658 [0.378, 0.880] | 0 |
| 1024×1024×1537 | 1.180 [1.163, 1.189] | 14 | 1.171 [1.158, 1.181] | 14 |
| 4096×4096×11008 | 1.097 [0.910, 1.334] | 10 | 1.124 [0.703, 1.400] | 12 |
| 8192³ | 1.075 [0.796, 1.745] | 8 | 1.182 [0.848, 1.497] | 12 |

### Seven-route end-to-end batch time

Separately measured warm host-wall microseconds, including dispatch through
completion; not a subtraction from the GPU phase.

| M×N×K | Native µs | TIRx µs | Hand µs | MPS µs | Torch µs | MPP µs | Views µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128×128×61 | 9.660 | 13.561 | 8.181 | 10.098 | 29.177 | 13.262 | 8.845 |
| 1024×1024×1537 | 512.048 | 1277.968 | 491.592 | 445.931 | 456.918 | 1188.600 | 521.213 |
| 4096×4096×11008 | 74207.958 | 74431.187 | 73260.834 | 51991.499 | 50808.562 | 75908.459 | 57790.291 |
| 8192³ | 417633.667 | 257562.688 | 379300.270 | 163072.813 | 170370.604 | 283810.583 | 193842.396 |

| M×N×K | E2E View/MPS [range] | Slower / 14 | E2E View/Torch [range] | Slower / 14 |
|---|---:|---:|---:|---:|
| 128×128×61 | 0.889 [0.767, 0.928] | 0 | 0.302 [0.291, 0.325] | 0 |
| 1024×1024×1537 | 1.169 [1.151, 1.184] | 14 | 1.141 [1.126, 1.150] | 14 |
| 4096×4096×11008 | 1.119 [0.946, 1.498] | 13 | 1.141 [0.549, 1.553] | 10 |
| 8192³ | 1.204 [0.948, 1.380] | 12 | 1.161 [0.957, 1.287] | 13 |

Single-call latency remains distinct. The following pairs are
`View/MPS; View/Torch`, with slower-round counts in parentheses. All four
metrics for every route, not only Views, are retained in [audit.json](audit.json).

| M×N×K | GPU single-call µs | GPU paired ratios (slower / 14) | E2E single-call µs | E2E paired ratios (slower / 14) |
|---|---:|---|---:|---|
| 128×128×61 | 10.792 | 0.913 (0); 0.495 (0) | 239.417 | 0.988 (6); 0.859 (0) |
| 1024×1024×1537 | 508.000 | 1.207 (14); 1.197 (14) | 753.146 | 1.136 (14); 1.048 (11) |
| 4096×4096×11008 | 59248.646 | 1.056 (11); 1.138 (11) | 59724.812 | 1.114 (11); 1.116 (12) |
| 8192³ | 216092.771 | 1.114 (10); 1.158 (13) | 210285.792 | 1.124 (11); 1.201 (12) |

The large-shape ranges are wide. In particular, the unchanged 8192³ view
control spans 192.820–285.891 ms across GPU round medians. Cross-session
differences and phase drift are not attributed to a compiler change,
temperature, cache or an inferred MPS kernel. The fixed K-tail view schedule
also has one host-throughput regression versus materialized MPP at
4096×4096×11008; it is retained, not excluded from the median.

## Implementation and numerical boundary

The optional C++ TVM patch adds `mpp_bounded_k_contract_version()==1` after
the existing memory contract v2, without replacing that ABI. The bridge
proves positive `actual_k=min(BK,source_k-origin_k)`, equal A/B lengths,
canonical zero guards, full M/N, unit logical projections, noalias and
immutable snapshot effects under enclosing execution domains. If guarded
forwarding cannot realize every reassociable MMA, the strict path is retried.
The actual-K tensors preserve physical strides and descriptor transposes.
Output ownership, pipeline ordering and accumulator proofs are unchanged.

This is **not** a same-schedule old/new speed comparison. The frozen old v2
binary rejects all three K-tail requests with `no legal Metal MPP group plan`;
there is no old execution time to divide by. The seven-path comparison keeps
the existing materialized 32×32×32 TIRx MPP control, so its ratio against
128×32×1024 views includes both view realization and geometry differences.
The aligned 8192³ view source retains exactly the historical identity
`b232075c58949157966874ef4a229d124b47e0df1c983804d22adb740c440ff5`.
It is an unchanged control, not a compiler improvement at that shape.

Regression validation after a full selected build:

| Check | Result |
|---|---|
| Complete Metal matrix suite, extended TVM | 1,857 assertions / 28 tests pass |
| Same current suite binary, frozen v2 TVM | 1,548 assertions / 28 registered tests pass; optional extension checks skip |
| CPU matrix / execution / pipeline / planner / targets | 5 selected CTests pass |
| Python benchmark contracts | 95 tests pass |
| Focused bounded-K numerical cases | 69 complete outputs pass |
| Frozen old binary's fixed K-tail requests | 3 admission rejections retained, not correctness passes |

The 69 outputs cover all A/B transposes, literal zero/nonzero memory C,
K=7/61/1033/11008, nominal BK=1024, two changed non-dyadic input sets and
pipeline window two. They use an independent FP64 oracle with
`atol=1e-4, rtol=2e-5`. Three semantic counterexamples preserve nonzero fill,
an extra mask and unequal effective K intervals. Typed ABI tests reject
zero/oversized extents, float actual-K and known insufficient strides.
The existing M/N-tail, alias, manual-memory, mutation and recurrence tests
remain in the full suite. No broader dtype or production LLM claim follows.

## Measurement contract

Four shapes × seven implementations × fourteen balanced orders; five samples,
20 ms target windows, 100 ms warmup and 300 s process timeouts. Shapes rotate
between rounds; every path occupies each position twice per shape and every
path pair has seven occurrences in each precedence order. The parameters
were fixed before timing. No concurrent build, test or profiler runs during
the replay. Tables use medians of within-round p50s; paired ratios are
medians of same-round ratios, not ratios of displayed medians. Ranges and
slower-round counts are descriptive, not confidence intervals.

FP32 `C=A×B`, compact row-major, alpha=1, beta=0, no transpose, prepacking,
returned-output allocation or reduced input precision. Native/handwritten
MPP disable fast math; TVM retains its existing Metal fast-math policy.
Torch uses eager `mm(..., out=...)` on MPS, without CPU fallback. Complete
outputs use the same deterministic dyadic inputs and FP64 oracle,
`atol=rtol=1e-4`; regression tests separately cover non-dyadic inputs.

The no-counter GPU controls sum command-buffer intervals, normalized per
invocation. They include GPU work and gaps, **not isolated shader instruction
time**. Instrumented compute-pass samples remain diagnostic raw evidence.
Host-wall batched throughput and synchronized single-call latency are
separate phases; do not subtract their medians from GPU medians to infer
dispatch overhead. Compilation, allocations, uploads/downloads and the
oracle are outside all timed phases.

## Provenance and reproduction

Machine: Apple M1 Max, macOS 26.6.2, Torch 2.14.0
(`08187d9e0fba026dc8217405802ab5381dc88d90`), NumPy 2.5.2; eight CPU threads
requested. This does not measure actual library worker use. The replay
records exact executable commands, source hashes, full output-validation
receipts, within-round samples and the linked-artifact inventory.

The source base is `8f56d9133c4f67f6fd287f95faffe40d914a3c5a` plus this
checkpoint's uncommitted changes at capture time. A pre-existing local
`cooperative.cpp` barrier-flag edit (3→2) is present in both frozen baseline
and candidate binaries but is **not submitted as a production source change**.
Two older source-string assertions still expect 3; neither was weakened and
no all-green dirty-worktree claim is made. The artifact/source hashes, not
the parent revision alone, identify these measurements.

Apply the optional patches and build using
[`patches/README.md`](../../../../../src/tile/bridge/tirx/patches/README.md).
The fixed replay command used this separate build:

```sh
env DYLD_LIBRARY_PATH=/tmp/luisa-tvm-mpp.VaKmzx/build/lib \
uv run --offline --no-project --python 3.13 --with numpy --with torch python \
  scripts/benchmark/tile_torch/compare_lowerings.py \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_native \
  --tirx /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_tirx \
  --mpp /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_mpp \
  --mps /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_system \
  --output scripts/benchmark/tile_torch/results/m1-max-20260906-mpp-bounded-k/replay \
  --shape 128,128,61 --shape 1024,1024,1537 \
  --shape 4096,4096,11008 --shape 8192,8192,8192 \
  --tirx-mpp --tirx-view-block 128,32,1024 \
  --metal-device-timing /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/libluisa-benchmark-metal-timing.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_compiler.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime_metal.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_runtime_extra.dylib \
  --compiler-artifact /tmp/luisa-tvm-mpp.VaKmzx/build/lib/libtvm_ffi.dylib \
  --rounds 14 --samples 5 --sample-ms 20 --warmup-ms 100 --timeout 300 --threads 8
```

Choose a new output directory to rerun; the orchestrator will not overwrite
this evidence. Run `audit.py --current-artifacts --self-test` while the frozen
binaries still exist; omit `--current-artifacts` later to audit the saved
receipts/sources rather than rebuilt binaries. Output arrays were checked
in full during execution but are not saved for independent recomputation.

## Documentation validation

This report stays in the existing repository structure: implementation
coverage, compiler matrix/runtime reference, validation record and performance
results. Exact lookup across shapes/routes is the reason for tables, rather
than a single-scale chart spanning microseconds to hundreds of milliseconds.
The compact proof diagram uses the same plain-text convention as its page.

The fresh strict Sphinx build retains ten pre-existing missing-Doxygen-XML
warnings; it is not a clean full-documentation build. The subsequent updated
pages build successfully. The local checker passes **48 HTML pages, 3,631
links/assets and 199 compatibility anchors**. No warnings were suppressed.

The [render check](docs_qa.cjs) verifies the actual HTML table against the
independent audit. Its [receipt](docs-qa.json) records 1440×1050 and 390×844
viewports. Desktop and mobile screenshots were inspected, including both
horizontal-scroll ends of the mobile table. The proof diagram fits the
narrow column, all metric columns remain accessible, and neither page
overflows the viewport. An existing long qualified option name was reworded
after the check exposed a 16-pixel mobile overflow; no site-wide CSS change
or separate report runtime was added.

## Remaining work

M/N-tail view admission, local K chunking versus whole-K atoms, explicit reuse
and participation/distribution families remain open. The current cost score
still charges nominal K conservatively. This checkpoint does not change its
coefficients or install per-shape search minima. Direct XIR/SIMD still needs
cache/register blocking and a matrix realization family; the earlier packet
proof is separate CPU evidence, not an MPS comparison.

The 8192³ MPS capture from the prior checkpoint remains locally available,
but its Xcode launch/counter attribution is unresolved. No old 1024³ capture
or cross-session timing drift is treated as evidence of the large kernel's
internal implementation. Further capture/profiling belongs outside acceptance
timing, with full output validation and a matching no-counter control.
