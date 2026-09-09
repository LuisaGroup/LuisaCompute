# Legacy migration and low-precision checkpoint

**September 9, 2026: the new design is merged into `next`, and development
continues there. FP16/BF16 now execute through XIR/SIMD and Metal/TIRx.
The expanded legacy comparison retains 38 shapes/operator cases across 191
requested route combinations, including every Error. The default-path matrix
still has a major scan regression. A later opt-in XIR map-fusion experiment
improves aligned CPU scans by 18–21× against the same compiler with fusion off;
it is not a remeasurement against legacy E2E or Torch/MPS. The overall
performance goal is not complete.**

The exact merge `2d02721b89980cd9191976c64b05d00b2342a03d` combines upstream
`6e58928d8` and design checkpoint `cd58daa6f`. A clean, recursively pinned
source export passed the full configured build, 80 XIR/SIMD tests, 35 Tile
tests, the Metal codegen test and the packed-word Metal regression before
the migration overlay. Existing user edits were not used for that verification
or included in the merge. New tests use explicit task-file overlays on this
same source baseline. Raw receipts, attempts and diagnostic measurements live
in `scripts/benchmark/tile_torch/results/m1-max-20260909-legacy-precision/`.

With the low-precision/migration overlay, the full configured build and
**80 XIR/SIMD, 41 Tile and one Metal codegen test passed** without skipped
tests. The expanded benchmark uses `precision-matrix-v2`: this adds the
separately labelled full-K benchmark capture to the v1 harness, with unchanged
production code. Its targeted rebuild and all three type plus three migration
test registrations also passed. The final receipts are
`precision-matrix-regression.json` and `precision-matrix-v2.json`; these are
separate from intermediate receipts preserved in earlier pilot archives.

## What changed

- Fifteen old example families have current execution-first captures in
  `examples/compute/tile/kernels.h`, shared by tests and a reusable benchmark.
  Small, odd/ragged and aligned cases check complete outputs and guard regions.
- FP16/BF16 now pass through typed constants, views, copies, casts, MMA and
  Runtime buffer binding. Accumulator precision is explicit, independent of
  input/output storage. INT8/UINT8 can use INT32 accumulation.
- XIR represents BF16 as encoded 16-bit values and explicit FP32 conversion,
  arithmetic and nearest-even rounding. TIRx retains BF16 format information
  and native legalization, with explicit encoding at semantic rounding events.
  Fused BF16 scalar bindings preserve these bits rather than widening away
  the rounding. No backend is told that BF16 integer-bit arithmetic is numeric
  BF16 arithmetic.
- FP8 E4M3FN/E5M2 have exact format, bit-storage and host decode interfaces.
  Current XIR/SIMD and Metal/TIRx Runtime execution rejects them explicitly;
  this checkpoint does not advertise native FP8 instructions. See
  [the language contract](../../tile/values.md#storage-and-arithmetic-precision).

These are representation/legalization improvements, **not a calibrated
precision-aware cost model or a new matrix atom**. Native Metal MPP remains
FP32. Existing source-semantic, layout and resource validation stays enabled.

## A fair legacy baseline

The restored `backup_old_tile/` is not a complete buildable frontend. The
historical source at `ccdfcbebef7fa95431c988e1fcbdd87ffdce9fdc` has identical
example and lowering blobs to the restored backup:

| Source | Verified blob |
|---|---|
| `tile_to_kernel.cpp` | `7dd4c2b65bc277d8682f334f841f59d230281536` |
| `examples/compute/tile_bench.cpp` | `552a6c495402ee81455bf7a790dbf13164d1ba7e` |

`build_legacy_exporter.py` recursively exports these pinned Git/submodule
objects into a new directory without switching or updating dependency
checkouts. It leaves the original examples and lowering unchanged. The only
historical library instrumentation adds JSON schema, argument-usage and warp
metadata required by the current strict AST decoder.

For the expanded matrix, a separate generated example translation unit
parameterizes host-side dimensions only. Original per-operator block sizes,
thread counts, pipeline settings and operation bodies stay unchanged. The
sized exporter reproduces **byte-identical AST and launch dimensions for all
16 original benchmark cases**. This establishes equivalence at the original
shapes; it does not assert that the old implementation supports arbitrary
ragged shapes. Failed old kernels are recorded, not repaired.

~~~text
old C++ Tile → original tile_to_kernel → ordinary SIMT AST ─┐
                                                          ├→ SAME current backend / Runtime
new C++ Tile → TileIR → TIRx (Metal) or XIR (SIMD) ─────────┘
~~~

This isolates frontend/lowering differences from backend age. It is not a
claim about how the complete historical backend performed. Legacy tensor
lowering is disabled, software pipeline enabled, asynchronous copies disabled;
these choices are recorded, not silently selected from benchmark winners.

## Diagnostic pilots, not an acceptance leaderboard

The initial FP32 Metal pilot has one legacy→modern order and three samples.
The follow-up has two alternating orders, three samples per visit, a 10 ms
target batch and 50 ms warmup. Desktop coactivity/thermal steady state and
statistical confidence are not qualified. Do not aggregate these into a
Torch/MPS claim. All passing visits check every output and both guards before
and after timing; input and artifact hashes are retained.

Metal numbers below are **sum of compute-encoder GPU intervals per dispatch**,
not host wall time. The artifacts also retain uninstrumented command-buffer
controls, batched E2E and single-call E2E latency; counter overhead and GPU
gaps are different quantities. CPU numbers are explicitly **batched E2E host
time**, not isolated native-entry execution.

| Initial FP32 Metal pilot | Shape | Legacy GPU µs | Current GPU µs |
|---|---|---:|---:|
| Copy | 8192×512 | 53.56 | 56.06 |
| Add | 2048² | 86.49 | 76.25 |
| SAXPY | 8192×512 | 95.15 | 77.53 |
| RMSNorm | 2048×256 | 28.95 | 16.32 |
| Sum | 8192×512 | 25.72 | 27.17 |
| Max | 8192×512 | 25.46 | 33.25 |
| GEMM | 4096³ | 510,539 | 284,323 |

In this initial pilot, large GEMM is faster than the legacy reference but still
takes hundreds of milliseconds: that is not competitive-library parity. Copy/sum/max regressions
are retained. The initial full-width scan graphs exceeded Metal stack capacity;
legacy 72² transpose failed Metal pipeline creation, so no transpose speedup
ratio is available.

| Follow-up, two round medians | Shape | Legacy µs | Current µs | Scope |
|---|---|---:|---:|---|
| FP16 GEMM, FP32 accumulation | 512³ | 292.0–316.5 | 257.5–314.9 | GPU intervals |
| Inclusive cumsum | 1024×256 | 18.9–20.1 | 732.8–739.4 | GPU intervals |
| Inclusive cummax | 8192×512 | 316.6–327.8 | 2316.2–2407.5 | GPU intervals |
| RMSNorm | 2048×256 | 1059.2–1153.5 | 149.9–166.7 | SIMD batched E2E |
| Sum | 8192×512 | 1713.1–2173.9 | 430.5–527.5 | SIMD batched E2E |

The FP16 case uses the same 16×16×32 staging geometry, FP16 inputs/output and
FP32 accumulation as the original 512³ example. The final-source replay above
has small FP16 wins, while the earlier replay had **4–6% regressions**. These
desktop pilots do **not** establish a stable FP16 performance improvement.
Both cohorts are retained, not selected by outcome. The portable blocked scan
now fits and is correct, but remains roughly **37–39× / 7–8× slower** in the
final replay than the old cumsum/cummax realization (the earlier replay was
also much slower). CPU E2E improvements do not establish CPU native
entry superiority or default-path Torch parity.

The structured test inputs are exact binary fractions with periods 97/89.
An independent FP64 dot-product oracle uses those documented periods to check
every 4096³ GEMM output economically; separate nonperiod-aligned tests verify
the oracle against direct dots. Low-precision GEMM references are rounded to
the output format. This is not random-data or production-distribution accuracy
qualification. GEMM input quantization is exact for this distribution.

## Expanded matrix: coverage improved, scan still regresses

The completed `precision-matrix-v2` cohort is independent of the diagnostic
pilots above: **38 cases, 191 unique requested case/route combinations and
382 planned visits**. There are 282 passing visits, 50 Error visits and 50
second visits deliberately not attempted after the corresponding first Error.
All current and legacy artifact hashes remained unchanged during measurement.

Cases were declared before timing: small/ragged and large pointwise operators,
wide row reductions, two scan widths, three transposes, five FP32 GEMMs and
four FP16 GEMMs. Each route gets two alternating-order visits, five samples
per visit, a 10 ms target batch and 50 ms warmup. Processes run sequentially;
each process has a 60-second whole-process limit. These are desktop diagnostic
observations, not confidence intervals or a thermally controlled acceptance
experiment. There is **no MPS/Torch baseline in this cohort**.

The four baseline/current columns below use the original old per-operator
geometry and the documented current geometry: pointwise/transpose 16×16,
four rows per row kernel, 32-element scan chunks, Metal pipelined GEMM
16×16×32 and SIMD GEMM 2×2×4. The two frontends have the same inputs and
complete-output/guard checks, not necessarily identical execution schedules.
`K=1` is a bookkeeping placeholder for non-GEMM operations.

GPU intervals are normalized per dispatch by
`compute_ns / 1000 / repetitions` before taking each visit's median.
CPU columns remain synchronized batched host E2E times; **do not compare their
absolute values directly with GPU-only columns**. Raw single-call E2E,
batched E2E, compilation time and uninstrumented GPU controls are retained
rather than mixed into the GPU table. The ranges below span the two visit
medians; they are not min/max individual samples or statistical error bars.

The source record is `matrix-results.json` in the checkpoint artifact
directory, with its files indexed by `matrix-manifest.json`. The tables are
generated by `summarize_legacy_matrix.py`; no failed baseline is dropped,
repaired or assigned a speedup.

### Full baseline/current matrix

Cells are the range of the two per-visit medians in **µs**. Metal uses GPU
compute-encoder intervals; SIMD uses batched E2E host wall. `Error` never
removes the other implementation's successful measurement. Failure reasons
and timeouts are in the raw visit records; no Error receives a speedup ratio.

| Operation / M×N×K | Legacy Metal | Current Metal/TIRx | Legacy SIMD | Current XIR/SIMD |
|---|---:|---:|---:|---:|
| copy 17×65×1 | Error | 4.63–4.73 | Error | 52.91–53.21 |
| copy 8192×512×1 | 60.57–60.63 | 60.38–62.52 | 4,741.44–4,868.90 | 637.49–803.94 |
| add 17×65×1 | Error | 4.32–4.98 | Error | 81.95–85.17 |
| add 8192×512×1 | 150.29–151.48 | 144.28–146.26 | 10,262.92–10,691.71 | 1,085.47–1,412.96 |
| saxpy 17×65×1 | Error | 4.85–6.28 | Error | 83.21–83.78 |
| saxpy 8192×512×1 | 163.62–169.53 | 147.53–150.34 | 10,757.33–10,839.38 | 1,023.68–1,147.84 |
| clamp 17×65×1 | Error | 6.48–18.82 | Error | 55.72–55.74 |
| clamp 8192×512×1 | 65.27–67.34 | 65.54–66.63 | 8,621.42–9,678.21 | 668.11–675.97 |
| exp 17×65×1 | Error | 6.32–7.52 | Error | 58.70–59.05 |
| exp 8192×512×1 | 98.61–131.14 | 74.47–108.38 | 9,271.71–9,547.92 | 1,311.62–1,360.83 |
| rmsnorm 17×65×1 | Error | 17.63–19.19 | Error | 34.26–34.75 |
| rmsnorm 512×4096×1 | Error | 41.61–44.80 | 1,918.47–2,015.75 | 660.21–781.18 |
| sum 17×65×1 | Error | 6.13–6.21 | Error | 16.87–17.04 |
| sum 512×4096×1 | 16.40–16.96 | 15.70–25.50 | 434.09–481.37 | 340.31–393.91 |
| max 17×65×1 | Error | 5.68–10.08 | Error | 16.83–16.98 |
| max 512×4096×1 | 16.71–16.78 | 15.69–17.05 | 434.86–462.48 | 415.87–443.08 |
| min 17×65×1 | Error | 8.29–10.13 | Error | 16.96–17.09 |
| min 512×4096×1 | 17.13–17.76 | 15.77–16.21 | 408.43–427.11 | 397.73–428.17 |
| abssum 17×65×1 | Error | 6.33–7.97 | Error | 17.96–18.19 |
| abssum 512×4096×1 | 15.87–16.42 | 15.36–16.36 | 393.35–414.38 | 433.04–457.77 |
| absmax 17×65×1 | Error | 5.80–6.24 | Error | 17.83–18.01 |
| absmax 512×4096×1 | 17.79–17.82 | 16.10–18.09 | 420.26–474.87 | 456.27–463.71 |
| cumsum 8×32×1 | 6.48–7.31 | 55.89–59.90 | 5.19–5.24 | 32.41–32.61 |
| cumsum 128×1024×1 | Error | 1,207.84–1,208.99 | 189.14–190.27 | 5,016.21–5,027.40 |
| cummax 8×32×1 | 7.73–30.37 | 59.64–63.40 | 34.66–38.88 | 32.41–35.89 |
| cummax 128×1024×1 | 30.19–32.19 | 1,202.17–1,211.42 | 187.99–235.10 | 5,015.29–5,071.56 |
| transpose 16×32×1 | 7.12–28.65 | 5.18–6.32 | Error | 1.46–1.47 |
| transpose 64×64×1 | 7.91–8.58 | 6.18–6.72 | Error | 4.18–4.18 |
| transpose 72×72×1 | Error | 6.81–44.08 | Error | 116.08–118.74 |
| gemm 32×32×32 | Error | 5.44–7.29 | Error | 43.92–45.11 |
| gemm 127×193×61 | Error | 15.49–15.90 | Error | 395.06–608.34 |
| gemm 512×512×512 | 1,340.03–1,449.93 | 109.01–128.90 | Error | 4,156.42–4,855.62 |
| gemm 1024×2048×256 | 5,832.79–5,955.12 | 422.56–491.53 | Error | 17,206.96–18,665.00 |
| gemm 4096×4096×4096 | 568,588.25–760,287.54 | 77,674.38–171,129.83 | Error | Error |
| gemm_fp16 32×32×32 | 7.62–8.75 | 7.48–10.77 | 60.58–71.91 | 55.67–78.85 |
| gemm_fp16 127×193×61 | Error | 13.49–21.46 | Error | 508.04–674.69 |
| gemm_fp16 512×512×512 | 323.80–335.13 | 328.48–345.42 | 30,689.79–35,926.00 | 7,892.00–9,992.83 |
| gemm_fp16 1024×1024×1024 | 1,874.20–2,302.22 | 2,133.64–2,512.79 | 180,466.33–222,097.33 | 57,283.42–61,809.83 |

Large pointwise SIMD E2E times are substantially lower than the legacy
replay, but reductions are mixed: wide absolute-sum remains slower, and
absolute-max is not a clear improvement. Metal pointwise/reduction differences
are much smaller; overlapping visit ranges should not be presented as stable
wins. FP16 Metal GEMM at 512³ and 1024³ does not establish an improvement over
the old path, despite the smaller wins in an earlier pilot.

Scan is the largest reproducible algorithmic gap in this matrix. At 128×1024,
current Metal cummax takes 1,202–1,211 µs against 30–32 µs for the old path;
current SIMD cumsum/cummax take about 5,000 µs against roughly 189–235 µs.
The old Metal cumsum fails at this shape, so it has **no comparison ratio**.
The successful current result remains in the table. A portable correct scan
is therefore not sufficient evidence of an effective collective mapping.

The current XIR/SIMD 4096³ FP32 GEMM cell is Error because the **whole benchmark
process exceeded its 60-second budget**, including setup, compilation,
validation and measurement. It is not a measured 60-second kernel time. Legacy
failures include ragged output-guard writes, numerical mismatches, and Metal
pipeline/SIMD codegen failures; exact diagnostics are retained per visit.

### Same pipelined source: additional Metal routes

These rows retain the 16×16×32 K-pipeline capture. Native MPP does not
support this source structure; its `Error` cells are not replaced by the
different full-K program below. An MPP-enabled TIRx policy can retain a
non-MPP fallback for an ineligible dtype; inspect `realization` in the log.

| Operation / M×N×K | Native MPP | TIRx, MPP-enabled policy |
|---|---:|---:|
| gemm 32×32×32 | Error | 5.21–18.09 |
| gemm 127×193×61 | Error | 23.28–25.23 |
| gemm 512×512×512 | Error | 83.05–106.10 |
| gemm 1024×2048×256 | Error | 337.53–340.97 |
| gemm 4096×4096×4096 | Error | 77,675.50–114,786.00 |
| gemm_fp16 32×32×32 | Error | 8.38–12.58 |
| gemm_fp16 127×193×61 | Error | 12.71–13.20 |
| gemm_fp16 512×512×512 | Error | 321.49–336.21 |
| gemm_fp16 1024×1024×1024 | Error | 2,059.88–2,081.68 |

### Same full-K FP32 source: three Metal routes

This is a **separately staged 32×32 full-K capture**, not a planner rewrite
of the preceding pipeline. All three columns use the same capture and inputs.

| M×N×K | TIRx | Native MPP | TIRx/MPP |
|---|---:|---:|---:|
| 32×32×32 | 8.64–8.68 | 13.19–16.00 | 18.87–26.58 |
| 127×193×61 | 24.52–25.36 | 14.70–17.19 | 11.71–13.09 |
| 512×512×512 | 66.21–71.65 | 82.06–83.58 | 48.71–54.29 |
| 1024×2048×256 | 282.02–304.71 | 363.30–388.11 | 185.95–199.35 |
| 4096×4096×4096 | 80,534.38–111,531.04 | 94,209.79–130,598.92 | 36,448.92–40,473.25 |

On the same full-K source, TIRx/MPP is faster than the native MPP realization
for the three larger cases here, while the tiny 32³ case reverses that ordering.
This is a useful mapping/atom-selection experiment, not evidence that a
single route should always win. In particular, the 36–40 ms full-K TIRx/MPP
4096³ result does **not** demonstrate MPS/Torch parity.

### Route coverage

| Requested route | Both visits pass | Error |
|---|---:|---:|
| legacy-metal | 21 | 17 |
| legacy-simd | 18 | 20 |
| metal | 38 | 0 |
| metal-direct | 5 | 0 |
| metal-native | 0 | 12 |
| metal-native-direct | 5 | 0 |
| metal-tirx-mpp | 12 | 0 |
| metal-tirx-mpp-direct | 5 | 0 |
| simd | 37 | 1 |

Coverage counts **unique case/route combinations**, not individual visits.
The twelve MPP-policy probes include the nine pipelined GEMMs above plus
ragged copy, sum and RMSNorm. A requested MPP policy is not proof that MPP was
used: the archived `requested_lowering_route`, `source_schedule` and
`realization` distinguish the policy, source and selected implementation.
The fifteen full-K combinations are separate rows in the experiment, not
successful replacements for pipelined native-MPP Errors.

The next optimization work should make the successful decomposition a
compiler choice: model cooperative scan communication and temporary costs;
bound SIMD scalar expansion; and jointly choose matrix atoms, worker
decomposition and pipeline structure. The full-K capture is a controlled
probe for that work, **not an implemented general planner transformation**.
A subsequent acceptance run still needs the dedicated CPU native-entry and
MPS/Torch harnesses under a qualified measurement protocol.

### Post-matrix precision boundary hardening

A subsequent review found that the pinned TIRx BF16 legalizer can widen a
BF16 MMA accumulator's local storage and erase its per-step rounding. TIRx now
rejects that unqualified accumulator mode, matching XIR, and both Runtime
routes have a negative test. BF16 inputs with FP32 accumulation remain supported.
`precision-final-boundary.json` records a successful targeted rebuild and all
six type/migration CTests after this change. This is a separate source snapshot;
the matrix above remains the v2 measurement, not a remeasurement of the final
boundary patch. All measured low-precision GEMMs use FP32 accumulation.

### Upstream CUDA integration check

After checkpoint `1c914d827`, upstream `17b71d4b4` was merged as `14ec3b7e8`.
Its new device-artifact exporter needed an explicit `tvm::ffi::String`
conversion for `InspectSource`; that one-line compatibility fix restores the
configured build. The six type/migration CTests pass again on the integrated
source. The additional upstream host CUDA-artifact test **fails** because the
pinned TVMx process has no `tirx.intrinsics.cuda.header_generator` or
`tirx.intrinsics.cuda.get_codegen` registration. Its negative-option case
passes, but its three artifact-generation cases do not. The failure is retained,
not skipped or counted as CUDA validation. No NVIDIA execution was attempted.
This integration check is separate from all matrix timings above.

## Pure-entry follow-up: generic map fusion

The follow-up improves **aligned CPU cumsum/cummax by approximately 18–21×**
without changing the library scan algorithm or reassociating its arithmetic.
Ragged cumsum improves only about 15%, transpose about 11%, and the unchanged
copy/absolute-sum controls show measurement noise. This is an opt-in generic
Tile-to-XIR realization, not a new scan opcode, a tuned default, or a claim of
legacy/Torch/MPS parity.

### Measurement boundary and results

The comparison is **map fusion off versus on in the same current compiler**.
Both use strict FP32, W8, one CPU thread and the same captured inputs, complete
FP64 oracle and source schedule. The timer calls the actual captured ORC object;
it does not rebuild the kernel from exported LLVM. Runtime dispatch, Python,
JIT, caller allocation, copies and validation are outside the common C++ timer.
Launch-record resets, block traversal and any compiler-emitted libc/allocation
remain inside. This is **single-thread complete native-launch host wall time**,
not CPU cycles, an inner-loop-only metric or multithread Runtime throughput.

Each case uses two ABBA cycles: four visits per variant, seven samples per
visit, 100 ms warmup and 30 ms target samples. The first two time columns are
medians of the four visit medians in microseconds. The ratio column is the
median of four matched on/off ratios, followed by their range; it is not the
ratio of the two displayed aggregate medians, nor a confidence interval.

| Operation / rows×columns | Fusion off µs | Fusion on µs | Paired on/off: median [range] |
|---|---:|---:|---:|
| cumsum 8×32 | 32.540 | 1.531 | 0.0471 [0.0466–0.0473] |
| cumsum 128×1024 | 5,050.552 | 256.593 | 0.0508 [0.0506–0.0512] |
| cummax 128×1024 | 5,030.826 | 268.644 | 0.0535 [0.0531–0.0537] |
| cumsum 17×65 | 131.444 | 111.591 | 0.8490 [0.8420–0.8582] |
| cumsum 1024×4096 | 161,096.042 | 8,841.563 | 0.0549 [0.0542–0.0555] |
| transpose 65×129 | 213.558 | 190.655 | 0.8926 [0.8902–0.8958] |
| abssum 17×65, unchanged control | 17.774 | 17.814 | 0.9986 [0.9924–1.0096] |
| copy 512×4096, unchanged control | 2,031.892 | 2,020.952 | 1.0216 [0.9690–1.0278] |

All 64 visits check every output and guards; outputs are byte-identical between
variants. An independent audit also recomputes all 6,563,363 unique case-output
elements from the structured inputs. Copy and absolute-sum have byte-identical
LLVM and objects across variants and `deferred_maps=0`: their apparent movement
is a control for noise, not an optimization result. These finite, structured
inputs do not qualify NaN, signed-zero or arbitrary production distributions.

Four aligned scan pairs change the compiler-generated entry from
`packet_batch` to `packet_batch.blocks`. The common adapter correctly executes
both, but the result includes this entry/traversal improvement. It must not be
described as a same-ABI, single-inner-body-call comparison. The desktop was not
thermally controlled or affinity-pinned; no other benchmark/build/profile was
run concurrently by this task during the timed replay.

### Why it helps, and what the planner still misses

Single-use, pure scalar maps and their index expressions become deferred
recipes evaluated at their consumer. Each recipe captures existing SSA
representations and coordinate ranges; it never delays an external memory load
past a write. Multi-use scan arithmetic stages retain their snapshots, so a
chain does not turn into exponential recomputation. Region/stage boundaries,
effectful maps and unsupported distributions keep the existing realization.
The current admission is limited to `local_lanes=1` and bounded recipe depth.

For the aligned scans, worker-private arrays fall from 17 to 7 and Schedule
blocks from 50 to 35; direct CFG becomes available. The ragged scan falls from
54 to 39 blocks but still uses the general scheduler. This explains why merely
counting removed arrays is insufficient to predict the observed speedup.

The planner shares the lowering admission and charges deferred work at actual
consumer reads, including repeated/broadcast reads. Its relative work prior
does **not** yet price the direct-CFG transition accurately or automatically
choose fusion on/off. The gain is a general compiler representation improvement
with matching work accounting, **not a calibrated cost-model victory**.
Enable the experimental realization with `PlannerOptions::enable_map_fusion`
or `LUISA_SIMD_ENABLE_MAP_FUSION=1`; the explicit disable environment option
overrides it. The default remains off.

The evidence is archived separately under
`scripts/benchmark/tile_torch/results/m1-max-20260909-map-fusion/`.
The v2 objects supplied the timed replay. The v3 defensive range/depth changes
were recaptured for all 16 variants: LLVM, ORC objects and all input/oracle/output
files are byte-identical to v2. v3 also retains the complete compiler-source
overlay and hashes all 24 relevant binary/plugin files before and after capture.
The original v2 binary inventory omitted `.so` plugins; its unchanged-binary
flag proves only the listed files, not a retrospectively complete inventory.
The exact measured ORC objects and replay libraries are independently frozen.

The final v4 patch aligns the planner/lowering recipe-depth boundary: 63 and
64 deferred levels succeed, while 65 and 70 fail closed with fusion enabled.
A complete configured build and six XIR/Runtime/LLM/type/migration CTests pass.
All 16 variants were recaptured again: the 16 LLVM files, 16 ORC objects and
50 input/oracle/output files remain byte-identical to v2. The v4 source overlay
and 24-file binary inventory are retained separately; this finite equivalence
check is not a fresh timing cohort or proof for untested kernels.

The next questions are whether ragged CFG can use a similarly efficient
realization, how ownership and communication planning can recover Metal scan
performance, and when recipe materialization should be selected automatically.
The legacy CPU route and Torch still need same-boundary native-entry replays;
the new scan times must not replace the old matrix's E2E column.

To replay an inspected migrated FP32 capture, use new output directories:

~~~sh
python3 scripts/benchmark/tile_torch/native_tile.py prepare \
  --prefix /path/to/off --log /path/to/off.log --objects /path/to/off-objects \
  --output /path/to/prepared-off --name off
python3 scripts/benchmark/tile_torch/native_tile.py prepare \
  --prefix /path/to/on --log /path/to/on.log --objects /path/to/on-objects \
  --output /path/to/prepared-on --name on
python3 scripts/benchmark/tile_torch/native_tile.py replay \
  --prepared /path/to/prepared-off/prepared.json \
  --prepared /path/to/prepared-on/prepared.json --output /path/to/replay
~~~

Preparation currently supports inspected Darwin arm64 buffer-only packet
entries, not cooperative kernels, aliased buffers or arbitrary scalar/resource
arguments. The helper's validation-only tests cover both entry ABIs, XYZ tails,
multiple buffers, complete-output errors, immutable inputs and guards.

## SIMD GEMM diagnosis and root traversal checkpoint

The current 4096³ SIMD Error is **not yet fixed in the default matrix route**.
An extended diagnostic establishes that compilation took about 34 ms and the
complete numerical/guard checks pass, but the original 2×2×4 kernel takes
about 13 seconds per execution. Four fixed warmup calls alone took about
53 seconds. The stack sample places workers in generated kernel code, not
compilation. It does not identify cache/TLB counters or prove a specific
memory bottleneck.

The XIR bridge now accepts a fixed, opt-in `root_axis_tiles` constraint, also
exposed as `LUISA_SIMD_ROOT_AXIS_TILES=32,32`. It normalizes a mixed-radix
root traversal, checks a static divisor/permutation/volume contract, and leaves
the source program's inner serial/pipeline/fold/MMA order unchanged. Equivalent
identity factorizations retain exact XIR/LLVM. The planner accounts for the
actual fastest digit and conservative cross-digit gathers, but **does not
search root factors or award temporal-cache credit**.

The following are separate single-sample, eight-worker Runtime host-wall
diagnostics, not ABBA, pure-entry timing or a replacement for the original
60-second-budget matrix. All use strict FP32; all 16,777,216 outputs and guards
are checked twice, with zero error on the structured inputs.

| 4096³ source microblock / realization | Packet width | Throughput sample, seconds |
|---|---:|---:|
| 2×2×4, original traversal | 8 | 13.111 |
| 2×2×4, full-packet specialization | 8 | 10.171 |
| 2×2×4, fixed root tiles 32×32 | 8 | 8.753 |
| Hand-staged 8×1×4 | 8 | 4.656 |
| Hand-staged 8×1×4, full-packet specialization | 8 | 2.570 |
| Hand-staged 8×1×4, full-packet specialization | 4 | 3.478 |

Root blocking improves this probe but its whole process still takes 70.38 s;
it does not close the timeout issue. The 8×1×4 probes also change per-program
register/load structure, so their gains cannot be attributed to root traversal
or automatic planner selection. A separate 1024×2048×256 root-blocked probe
passes with a 17.782 ms throughput sample; this is not a regression qualification
across sizes or operators.

The complete configured build, three XIR/Runtime/LLM CTests and five changed-C++
syntax checks pass. New tests check the emitted address sequence and bijection,
identity code, invalid factors, noncommutative recurrences and ragged local-lane
reduction. The final source overlay and all seven diagnostic captures are in
`scripts/benchmark/tile_torch/results/m1-max-20260909-simd-root-traversal/`.
No pre-run full binary inventory or captured ORC object was recorded for these
GEMM probes; retained source/output evidence must not be overstated as binary
provenance. The original Error remains unchanged.

The next work is a generic per-resource access/reuse analysis plus measured
issue/cache/TLB costs, followed by multi-size pure-entry and Runtime replay.
The Chinese implementation proposal and overnight handoff are
{download}`root-mapping cost notes <../../../../src/tile/ROOT_MAPPING_COST_NOTES.zh.md>`
and {download}`September 9 handoff <../../../../src/tile/HANDOFF_2026-09-09.zh.md>`.
CUDA lowering is the overnight priority; SIMD optimization pauses at this
checkpoint, without claiming the overall Torch/MPS/BLAS performance goal.

## Failures retained and what they teach

1. **Numerical boundaries matter under fusion.** The first BF16 round-trip
   test exposed promotion erasing an explicit narrowing event. Keeping the
   output buffer correct was insufficient. The corrected bit-level contract
   now checks both stored BF16 and the value widened back to FP32. Typed fused
   bindings also work around a pinned TVMx BF16 Bind retyping defect without
   modifying the dependency checkout.
2. **Physical storage must match metadata.** The Runtime's ordinary structure
   layout has a minimum four-byte alignment. BF16/FP8 wrappers therefore use
   exact-size one-element storage arrays in the Runtime adapter; Tile argument
   metadata retains the numerical format. Tests check actual stride and byte
   counts, not just `sizeof` on the host.
3. **FP8 enum support is not codegen support.** Attempting both FP8 copies
   reached invalid Metal `as_type<half>` expressions after C integer promotion.
   Their failed attempts are retained. The current Runtime gate now explains
   that FP8 is unqualified rather than invoking a known-invalid path. XIR has
   its own missing-legalizer diagnostic. These are negative tests, not claimed
   FP8 kernel successes.
4. **SSA size is an optimization constraint.** A 4×8×8 SIMD GEMM correctness
   case exceeded the test timeout while LLVM scheduled machine instructions.
   A one-second sample was entirely in machine scheduling/register-pressure
   accounting, not kernel execution. The routine test uses 2×2×4; the compile
   cliff remains evidence for a bounded representation/atom plan, not a
   fabricated kernel-time result.
5. **Feasibility is not profitability.** Blocking the library scan bounds
   temporary size and fixes the stack failure. It does not synthesize a
   cooperative prefix scan or select its worker-axis/resource mapping.
6. **Measurement guards must fail closed.** The first CPU pilot accidentally
   inherited the Metal timestamp setting and was rejected. Those eight failed
   harness visits remain in the archive; a clean CPU-only replay supplies the
   E2E values, followed by the final-source replay above. They were not backend
   failures or discarded slow samples.

The follow-up priority is consequently general: preserve numerical events in
fusion; price dtype-specific conversion/atom costs; bound SSA expansion and
register pressure; and select lane/worker decomposition plus communication for
scan/reduction patterns. Choosing another hand-tuned operator name or claiming
the newly feasible scan is fast would not address these gaps.

## Reproduce

Build the current CMake test/benchmark targets with the required SIMD and/or
Metal + pinned TIRx configuration, then run:

~~~sh
ctest --test-dir build -R '^test_tile_(types|migrated)(_simd|_metal)?$' --output-on-failure
python3 scripts/benchmark/tile_torch/build_legacy_exporter.py --output /tmp/new-legacy-export
/tmp/new-legacy-export/build/bin/emit_legacy_tile gemm_fp16 /tmp/old-gemm-fp16
build/bin/benchmark_tile_migrated metal gemm_fp16 512 512 512 16 16 32 3 10 50 /tmp/current-fp16
build/bin/benchmark_tile_migrated metal gemm_fp16 512 512 512 16 16 32 3 10 50 /tmp/legacy-fp16 /tmp/old-gemm-fp16.ast.json 8192 32
~~~

To reproduce the expanded cohort, first verify the sized exporter and provide
the receipt for the exact current source overlay/build. The output and
identity directories must not already exist:

~~~sh
python3 scripts/benchmark/tile_torch/verify_legacy_exporter.py \
    --bin-dir /tmp/new-legacy-export/build/bin --output /tmp/new-legacy-identity
python3 scripts/benchmark/tile_torch/run_legacy_matrix.py \
    --binary build/bin/benchmark_tile_migrated \
    --exporter /tmp/new-legacy-export/build/bin/emit_legacy_tile_sized \
    --source-receipt /path/to/current-source-receipt.json \
    --legacy-provenance /tmp/new-legacy-export/provenance.json \
    --output /tmp/new-legacy-matrix --rounds 2 --samples 5 \
    --sample-ms 10 --warmup-ms 50 --timeout 60
python3 scripts/benchmark/tile_torch/summarize_legacy_matrix.py /tmp/new-legacy-matrix/results.json
~~~

The runner resolves its Metal timing library next to the benchmark binary,
removes inherited tuning overrides, records the complete planned cohort and
retains every failure. It does not tune or repair legacy kernels. Consult the
artifact directory's README for archived evidence and source reconstruction.

All output prefixes must be new. Set `LUISA_TILE_BENCH_METAL_TIMING` to the
built timing library only for Metal GPU measurements; do not pass it to CPU
runs when invoking the benchmark directly. `gemm_bf16` runs the current BF16
counterpart, with no claimed historical
BF16 baseline. Integer GEMMs are correctness tests, not this FP32-accumulation
benchmark. CPU native-entry and MPS/Torch acceptance cohorts still need their
existing dedicated measurement harnesses.
