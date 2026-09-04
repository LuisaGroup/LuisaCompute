# Tile programming: implementation and evidence report

As of September 5, 2026, Asia/Shanghai. This is a technical status report for
the `codex/tile-programming-design` branch and the source/report snapshot that
contains it. A Git revision alone does not reproduce performance experiments:
use the recorded binary/source hashes and exact commands in the linked
evidence.

## Technical summary

The packaged [evidence report](../../scripts/benchmark/tile_torch/results/m1-max-20260905-xir-simd/report.html)
contains the same bounded snapshot, charts, audit tables and canonical sources.
Its canonical validator and structural verifier pass. The packager found no
compatible Chromium headless-shell, so responsive/source-dialog browser QA is
not claimed; the self-contained semantic chart and table fallbacks remain in
the file. This Markdown page remains the repository-native detailed report.

The execution-first C++ Tile language now has three distinct working compiler
routes: the maintained C++ TIRx bridge (including a typed Metal MPP contract),
a limited native Metal MPP realization, and a direct TileIR→XIR→SIMD CPU
realization on ordinary Luisa Runtime.
The XIR route includes a real, bounded execution-map solver; it does not
mechanically preserve the root axis order. Its model is still an uncalibrated
prior over a narrow candidate space, not a solved general CPU scheduler.

Correctness coverage now extends beyond GEMM to normalization, activations,
RoPE, masked softmax and online causal prefill/decode/GQA. The same captured
program is tested through XIR and TIRx against an independent FP64 reference.
Those are small multi-shape correctness cases, not production-size LLM
performance claims.

Metal MPP contract v2 now recognizes a proved positive-zero, single-iteration
accumulator as overwriting `D=A*B`. It removes the generated cooperative-tensor
zero-fill and C input instead of asking MPP to multiply-accumulate into zero.
MPP cost-model v2 now ranks legal rectangles by subgroup critical-path work
and whole-device subgroup waves. On the finite 8-shape calibration cohort it
reduces mean model regret from 74.18% to 8.82%; this is in-cohort evidence, not
held-out calibration. In the independent 14-round replay, the selected TIRx
MPP-view plans beat Torch and MPS on all eight GEMMs. At 1024³ they are 4.87%
faster than Torch and 0.62% faster than MPS by paired time ratio.

The TIRx CPU route now also has two explicit, structurally proved provider
families. A whole compact FP32 `C=A*B` contract can become one CBLAS call;
shared compact FP32 exponentials and exact add/max/min recurrences can become
Accelerate array operations. A six-order GEMM replay is faster than eager
Torch on seven of eight shapes and within 0.5--10.5% of direct CBLAS on seven
of eight; the 32³ wrapper-dominated case is 25.4% slower. A separate six-round
policy A/B shows no systematic add change, 2.71--6.12× reduction speedups and
2.10--5.46× softmax speedups. All 192 replayed outputs pass their complete
oracles and the fingerprinted artifacts remain unchanged.

**The general library-performance goal is still not complete.** These CPU
wins are legal provider realizations for narrow proved contracts, not evidence
that the portable loop family or direct XIR route has acquired BLAS-class
matrix and vector-math atoms. Direct XIR GEMM remains 38--55× behind eager
Torch in its pilot; Metal has not been benchmarked across a production LLM
operator suite; and the MPP model has no held-out device/operator validation.
No Metal result is relabeled as XIR performance.

## Reading map and scope

| Document / artifact | What it answers |
|---|---|
| [Language and layout design](tile_programming_design.md) | Minimal primitives, lexical Nests, assignment capture, layout algebra and proof boundaries |
| [Executable kernel gallery](tile_programming_poc_kernels.md) | GEMM, reductions, attention, CNN/filter and Top-K/sort syntax/composition examples |
| [Execution planner](tile_execution_planner.md) | General binding/distribution/atom/resource/time formulation; current Metal implementation versus future calibrated model |
| [Runtime and native lowering](tile_native_runtime.md) | Factory, backend ownership, shader handles, Metal native/TIRx limits and ABI |
| [XIR execution planning](tile_xir_design.md) | Exact implemented CPU candidate space, cost equations, alias constraints, SSA/CFG lowering and extension roadmap |
| {download}`Benchmark tools <../../scripts/benchmark/tile_torch/README.md>` | Reproduction interfaces, timing definitions, controls and saved runs |
| {download}`Validation evidence <../../scripts/benchmark/tile_torch/results/m1-max-20260905-xir-validation/notes.md>` | Test commands, failure investigations, logs and audit details |
| {download}`Metal MPP cost v2 search <../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-search/notes.md>` | v1 failure, v2 equations, hard legality and in-cohort model regret |
| {download}`Metal MPP cost v2 replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-replay/notes.md>` | Frozen schedules, 784 full-output checks and balanced MPP/MPS/Torch evidence |
| {download}`CPU CBLAS replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-replay/notes.md>` | Eight frozen GEMMs, six implementation orders, direct CBLAS overhead and Torch comparison |
| {download}`CPU array-math replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-accelerate-ops-replay/notes.md>` | Causal reference/Accelerate A/B over add controls, row reductions and softmax |
| {download}`Final CPU/provider validation <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-provider-validation/notes.md>` | Full rebuild, focused structural tests, 34/34 submitted-source Tile cohort, 64/64 Python checks and documentation QA |

```{figure} ../_static/tile/xir-planning-pipeline.svg
:alt: The same execution-first TileIR feeds XIR/SIMD, Metal-native MPP and TIRx compiler routes with separate ownership.
:width: 100%

One language and Runtime contract do not imply identical target schedules or interchangeable benchmark results.
```

## What is implemented, and what remains design

| Area | Implemented and exercised | Important remaining boundary |
|---|---|---|
| C++ surface | Signature parameters; range-for Nests; direct carried assignment; explicit stores; Tile-level operations | Not arbitrary C++ capture or intra-kernel SIMT/Tile mixing |
| Execution | `parallel`, `serial`, `pipeline`, `reduce`; scope constraints | The backend must realize the requested binding; unsupported bindings are errors |
| Data/layout | Typed layout representation and proof mechanisms; Tensor as storage plus layout/view | Not every represented layout has an emitter on every bridge |
| TileIR | Mutable typed SSA, regions and intrusive ownership/use structure | General Machine TileIR and its pass suite are not implemented |
| TIRx | Native C++ export, shared structural lowering, CPU/Metal realizations; typed MPP v2 modes and bounded target-specific cost/solver | Held-out calibration and broader atoms/operators remain necessary |
| Native Metal | Typed FP32 MMA/view-forwarding subset; ordinary Runtime shader and launch | Not general epilogues, K pipelines, manual Memory, all dtypes or arbitrary operators |
| XIR/SIMD | Direct verified XIR; local Tile expansion; loop PHIs; ordinary CPU Runtime | No matrix-extension atom, packed GEMM microkernel or general Tile distribution |
| CPU planner / realizations | Root-axis permutations × legal worker-block widths; bounded storage/SIMD/launch choices; proved CBLAS and Accelerate atoms | Provider selection is explicit; no fitted break-even model, whole-program optimum, general Tile partitioning or physical pipeline solver |
| Autotuning | Recapture/JIT variants and frozen-plan benchmarking | Broader search requires legal emitters and measured ranking; one capture is not mandatory |

The existing CuTe-derived mixed-radix/composition design is not a claim of a
complete decision procedure over arbitrary programs. The language design
distinguishes representational closure, proof fragments, finite fallback and
unknown results. Likewise, XIR's current compact-buffer realization is a
subset of the layout representation, not an alternative, less general DSL.

## Correctness: common LLM operators now use both bridges

`test_tile_xir_llm` runs **21 captured kernel/shape combinations**, each through
XIR/SIMD and native-target TIRx CPU, with `atol=rtol=5e-5` against independent
FP64 formulas. Every output element is checked. XIR outputs begin as NaNs and
use an offset BufferView with guards before/after the writable range.

| Family | Shapes / edge cases | Reference |
|---|---|---|
| RMSNorm, LayerNorm | 17 rows × widths 7, 32, 65; shared gamma/beta | FP64 mean/variance, epsilon and affine transform |
| SwiGLU, GELU+residual | Same three widths; non-dyadic inputs | FP64 sigmoid/tanh formulas with the same float coefficients |
| Masked softmax | Same widths; row-dependent nonempty mask | Stable FP64 masked exponential normalization |
| Split-half RoPE | 17 rows × widths 6, 32, 66 | FP64 rotation using identical supplied sine/cosine tables |
| Online attention | `(B,Hq,Hkv,Q,K,D,Dv)` = `(1,2,2,4,5,4,3)`, `(2,4,2,7,11,8,7)`, `(2,4,2,1,17,8,7)` | Full-score FP64 causal softmax and value contraction, independent of the online recurrence |

Attention queries represent the final Q positions of the KV sequence. Local
query/key tiles are 2×3, so these cases exercise tails, causal masks, online
max/sum/accumulator carries and grouped query heads. These tests do not measure
KV-cache paging, variable-length batches, long contexts or production hidden
dimensions. CNN, traditional filters, Top-K and sort remain available in the
language/earlier TIRx gallery; they are **not newly validated on XIR** here.

Additional XIR tests cover transposed/ragged GEMM, nonzero accumulators, two
changed non-dyadic input sets, loop-carried swaps, zero-trip loops, view
offsets, read/write snapshots, move-only shader lifetime, negative origins and
signed-overflow rejection in the bounds proof. The dedicated SIMD PHI test
uses widths 1/2/4/8/16 and every active-lane count, independent of TileIR.

The submitted source preserves `metal::mem_flags(3)` and the final complete
`test_tile_*` regression cohort is **34/34 passing**: all 31 `unit_tile` tests
plus the three separately registered XIR/LLM/native Runtime integrations.
For ownership auditing, the same cohort was first run against an unowned local
`mem_flags(2)` edit and reported 32/34; the only two failures were the explicit
generated-source assertions in `test_tile_tirx_cooperative_metal` and
`test_tile_tirx_memory_metal`. Restoring the submitted value made the full
31-test label pass without weakening either assertion. The local edit is not
part of this source snapshot. The Python benchmark-contract suite passes
**64/64**. See the
{download}`final validation note <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-provider-validation/notes.md>`
for commands and scope; this is not a claim that every non-Tile repository
test passed.

## Three failure investigations changed the implementation

### LLVM coexistence is a build/runtime constraint

The first combined TIRx/SIMD process loaded LLVM 21 through TVM and LLVM 22
through SIMD, then crashed in LLVM analysis setup before kernel execution.
Configuring both stacks against LLVM 21.1.8 removed that crash. An XIR-only
LLVM-22 executable had worked, which helped isolate the combined-process
configuration. This does not mean LLVM 22 is intrinsically unsupported; the
tested combined stack uses matching versions.

### PHI transfers must be simultaneous

The Tile bridge emitted valid loop PHIs, but SIMD's edge assignment lowering
loaded/stored one assignment at a time. A cycle `a←b, b←a` could read an
already overwritten state slot. The fix snapshots all right-hand sides
before any destination update. A pure-XIR regression produced 62 failures
before the fix and passed afterwards, along with existing SIMD regression
tests. Slot-coloring policy was not relaxed as part of this repair.

### Bounds proofs belong before Schedule expansion

Initial LLM runs were stopped during excessive JIT work, not recorded as
numeric passes. A process sample located LLVM machine scheduling/register
pressure work. Shared SSA/CFG cleanup and avoiding an extra diagnostic
assembly compilation helped but were insufficient. Per-axis checked integer
range proofs now remove redundant bounds diamonds before XIR/Schedule
expansion; unknown/overflowing accesses retain the original guarded behavior.

The complete LLM test subsequently finished in approximately 12 seconds,
including capture, both JIT routes, launches and validation. This observation
is a compilation/verification usability result, **not a kernel speedup ratio**;
the interrupted runs are not comparable completed timing samples.

## Performance: preserve the measurement basis

Unless explicitly labeled otherwise, reported times are **warm synchronized
host-wall time per invocation**, amortized over a batch. They include each
runtime's dispatch/encoding/submission and synchronization. They exclude JIT,
allocation/upload and cold-call setup. They are not GPU hardware-event times,
and CPU thread requests are not measurements of actual library worker use.

Report tables use medians of within-round p50s. A paired ratio is the median
of same-round numerator/denominator ratios, **not** a ratio of the displayed
medians. Ranges and counts of slower rounds are descriptive, not confidence
intervals. No slow or failed row is discarded to improve the headline.

### New XIR/SIMD planner pilot

The {download}`XIR pilot <../../scripts/benchmark/tile_torch/results/m1-max-20260905-xir-simd/notes.md>`
compares automatic planning, fixed `{order=[0,1], block=64}`, and eager
Torch on the same CPU, separately from TIRx. The Tile specialization is fixed
at 1×1×8. Six rounds balance all three implementation orders for 32³, 128³
and 127×193×61. Raw outputs, LLVM source, actual plans and hashes are retained.

This pilot asks whether the initial mapping prior helps that specific
specialization; it is not a production GEMM schedule search or a benchmark
of the LLM family. The report preserves negative results and names the exact
comparison baseline. Neither a better model score nor a passing test is
reported as a speedup.

| Shape | Planned µs | Fixed map µs | Torch µs | Paired planned/fixed | Paired planned/Torch |
|---|---:|---:|---:|---:|---:|
| 32³ | 50.822 | 50.037 | 0.978 | 1.0157× | 51.851× |
| 128³ | 272.410 | 278.352 | 4.979 | 0.9755× | 54.543× |
| 127×193×61 | 255.711 | 281.315 | 6.696 | 0.9142× | 38.186× |

All automatic plans chose root order `[0,1]`; worker packing differed from the
fixed 64-worker control. Automatic planning was slower in 4/6, 3/6 and 1/6
rounds respectively. The fixed comparison therefore shows a modest, noisy
mapping effect, while the 38–55× Torch gap diagnoses a missing realization
family. This is direct evidence for Tile/lane distribution, register blocking,
cache-aware reuse and vector/matrix microkernels before cost-model polishing.

### Balanced Metal evidence: MPP cost v2 closes this GEMM cohort

The {download}`cost-model study <../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-search/notes.md>`
first preserves the failed v1 ranking. Across the same 8 shapes and 45 requested
block/thread candidates, v1's mean/median/maximum finite-set regret is
74.18/43.05/239.58%; v2's is 8.82/2.59/34.37%. Exact measured-winner picks
increase from 1/8 to 4/8. Those 3-sample, 10 ms values are **in-cohort** and
diagnostic. They neither establish held-out prediction nor replace final timing.

The independent {download}`v2 replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-replay/notes.md>`
freezes the measured schedules, then uses 14 balanced rounds, 8 shapes and
7 compiler/library paths. All 784 complete outputs passed the same FP64 oracle;
all 21 fingerprinted benchmark/compiler/runtime artifacts retained their hashes.
No schedule was searched or selected during replay.

```{figure} ../_static/tile/mpp-cost-model.svg
:alt: MPP planning generates and proves legal candidates, applies a target-specific relative-work model, searches the bounded space, and finally defers to correctness-checked JIT measurement.
:width: 100%

The analytic plan is a shortlist prior. The independently validated measured winner is what the replay freezes.
```

| Shape | Frozen block @ threads | TIRx MPP views | Hand MPP | MPS | Torch | Paired view/MPS | Paired view/Torch |
|---|---|---:|---:|---:|---:|---:|---:|
| 32³ | 32×32×32 @ 128t | 2.982 | 2.809 | 10.081 | 26.899 | 0.2794× | 0.1105× |
| 128³ | 32×32×128 @ 256t | 5.335 | 5.441 | 16.904 | 27.218 | 0.3174× | 0.1943× |
| 512³ | 32×64×32 @ 128t | 42.413 | 46.802 | 52.428 | 47.745 | 0.8285× | 0.8919× |
| 1024³ | 128×32×1024 @ 128t | 270.675 | 266.105 | 272.572 | 284.654 | 0.9938× | 0.9513× |
| 256×1024×128 | 64×64×128 @ 256t | 16.025 | 17.286 | 20.350 | 28.668 | 0.8189× | 0.5554× |
| 1024×128×256 | 32×32×32 @ 128t | 16.500 | 18.508 | 26.270 | 28.655 | 0.5946× | 0.5596× |
| 127×193×61 | 32×32×32 @ 256t | 8.861 | 7.127 | 16.915 | 26.997 | 0.5172× | 0.3266× |
| 513×257×129 | 32×32×32 @ 256t | 20.607 | 24.424 | 35.043 | 34.002 | 0.5874× | 0.6057× |

At 1024³ the new 128×32×1024, 4×1-subgroup schedule is 4.87% faster than
Torch, 5.76% faster than native Tile→MPP and 0.62% faster than MPS by paired
ratio; it remains 1.68% slower than handwritten MPP. It was slower than MPS in
only 1/14 rounds. Across all eight rows the TIRx-view path beats both external
baselines. This closes the measured FP32 GEMM cohort, not the general library-
performance goal: model v2 still needs held-out shapes/operators and residual
regret shows that cache/layout, edge and launch features are incomplete.

Native/handwritten MPP use fast math off; TVM's Metal runtime uses fast math
on. All values above are synchronized host-wall batched times, not GPU-event
durations. The original TIRx, staged TIRx MPP, native MPP, handwritten MPP,
MPS and Torch controls remain in the raw report.

### CPU TIRx: reference gaps and proved provider realizations

The original six-round, eight-shape reference-loop cohort remains useful as a
negative control. At 1024³ it measured 5919.062 µs versus Torch at 1020.527 µs
and direct Accelerate at 1027.681 µs: a paired 5.769× TIRx/Torch gap. Changing
only `target-cpu` did not change the emitted 4×16 register-blocked loop body or
close that gap. Cache-aware panels, packing/reuse and a matrix microkernel were
absent from the reachable realization family, so a better solver score could
not help.

The new solution preserves the same TileIR semantics but adds a target
realization boundary. Structural TileIR matching proves a whole compact FP32
GEMM or an exact reduction recurrence; shared expensive Tile SSA is
materialized once before export. The CPU pass then revalidates the actual TIRx
body, buffer ABI, layout, alias contract and target policy before replacing it
with a provider atom. It never matches a diagnostic operation name, and an
explicit unsupported request fails rather than silently changing semantics.

```{figure} ../_static/tile/tirx-realization-pipeline.svg
:alt: TileIR is structurally exported once, then portable, CPU-provider and Metal matrix families are selected behind a second proof firewall.
:width: 100%

Provider calls are target realizations selected from proved semantic
contracts; direct CBLAS/MPS benchmark programs remain independent baselines.
```

#### Whole-GEMM CBLAS realization

The {download}`current single-session plan <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-plan/notes.md>`
verifies that each generated LLVM kernel has exactly one external matrix call.
The {download}`six-order replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-replay/notes.md>`
then freezes those schedules and remeasures Tile, eager PyTorch and a separate
direct Accelerate CBLAS executable. There are 48 valid complete-output rows,
zero failures, and stable binary/library hashes.

| FP32 shape M×N×K | Tile→TIRx→CBLAS µs | eager Torch µs | direct CBLAS µs | paired Tile / CBLAS [range] |
|---|---:|---:|---:|---:|
| 32×32×32 | 0.503 | 0.918 | 0.390 | 1.254× [1.071, 1.484] |
| 128×128×128 | 4.518 | 4.961 | 4.073 | 1.105× [1.085, 1.148] |
| 512×512×512 | 130.099 | 139.469 | 131.055 | 0.995× [0.988, 1.002] |
| 1024×1024×1024 | 984.515 | 930.311 | 965.743 | 1.031× [0.893, 1.234] |
| 256×1024×128 | 65.597 | 65.877 | 64.332 | 1.020× [1.007, 1.026] |
| 1024×128×256 | 62.717 | 63.152 | 61.323 | 1.023× [1.019, 1.028] |
| 127×193×61 | 6.287 | 6.791 | 6.030 | 1.047× [1.012, 1.075] |
| 513×257×129 | 43.612 | 43.701 | 43.356 | 1.005× [0.990, 1.035] |

The Tile path beats the displayed Torch median on seven of eight shapes. The
comparison against direct CBLAS answers a different question: wrapper and TVM
packed-ABI overhead are visible, especially at 32³ and 128³. The wide 1024³
range also shows why one lucky run must not be used as the headline.

#### Shared transcendental and reduction realization

The structural exporter now materializes a shared `exp` Tile once when its SSA
result has multiple consumers, instead of expanding the lazy expression into
both a reduction and an output consumer. The opt-in
`CpuMathBackend::ACCELERATE` policy can then realize that exact compact map with
vForce and exact FP32 add/max/min recurrence contracts with vDSP. The reference
path remains available. Unrelated add kernels are a negative control.

The {download}`six-round policy replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-accelerate-ops-replay/notes.md>`
contains 144 freshly captured/JIT-compiled rows, all valid. Times are medians
of per-round p50 synchronized host-wall measurements. The speedup is the median
of paired reference/candidate ratios, not a ratio of selected best runs.

| Case | Reference µs | Accelerate realization µs | Paired speedup [range] | candidate-run Torch µs |
|---|---:|---:|---:|---:|
| add 1×127 | 0.068 | 0.068 | 1.001× [0.978, 1.067] | 0.548 |
| add 17×257 | 0.421 | 0.418 | 1.001× [0.998, 1.018] | 0.934 |
| add 128×1024 | 4.698 | 4.713 | 1.004× [0.885, 1.226] | 38.289 |
| add 4096×256 | 32.807 | 32.207 | 1.022× [0.958, 1.437] | 84.070 |
| sum 1×127 | 0.064 | 0.024 | 2.708× [2.587, 2.774] | 0.772 |
| sum 17×257 | 2.186 | 0.375 | 5.828× [5.564, 5.939] | 1.060 |
| sum 128×1024 | 16.640 | 3.703 | 4.581× [3.534, 4.978] | 37.578 |
| sum 64×4096 | 33.738 | 5.512 | 6.123× [5.267, 7.228] | 40.651 |
| softmax 1×127 | 0.551 | 0.126 | 4.357× [4.276, 4.370] | 0.619 |
| softmax 17×257 | 5.436 | 2.555 | 2.098× [1.980, 2.286] | 33.428 |
| softmax 128×1024 | 79.242 | 14.527 | 5.460× [5.159, 5.609] | 88.699 |
| softmax 64×4096 | 156.785 | 41.876 | 3.753× [3.524, 4.113] | 128.818 |

The independent add control staying near 1× is evidence that the policy does
not broadly rewrite unrelated code. The single-session candidate report also
records zero provider calls for add, one dynamic reduction operation per row,
and three semantic call sites for softmax (`max`, `exp`, `sum`). Static LLVM
call-site counts can be larger when a small serial root is unrolled; they are
not dynamic-call counters.

This policy has a deliberately different numerical contract. vDSP may choose
a different FP32 reduction order, while vForce documents different denormal
and floating-exception behavior from scalar libm. The benchmark accepts it
only through the explicit target option and checks all outputs with recorded
tolerances; it is not silently enabled by the Tile DSL or execution hierarchy.

Two other CPU scheduling repairs matter independently of providers. Automatic
roots below 64 cheap tasks stay serial unless the source explicitly requests a
worker scope; small roots containing transcendental/opaque work retain
parallel execution. Ragged SIMD packs are binary-versioned into a proved
all-lanes fast arm and an unchanged guarded slow arm. This removed full-pack
store scalarization: the 17×257 add control is now about 0.42 µs instead of the
earlier 2.84 µs observation. Both policies preserve the original tail and
parallel semantics and have dedicated structural/numerical tests.

## Next work and acceptance criteria

1. **Generalize the CPU atom catalog:** add layout/stride/transpose and fused
   epilogue contracts without turning whole operators into DSL primitives.
   Select reference, library and native microkernel atoms with an explicit
   break-even model; preserve the current opt-in policies as controls.
2. **Close the direct XIR/reference gap:** choose Tile/vector axes, reduction
   trees, register blocking and cache/packing only with dependence, alias and
   numerical proofs. Provider parity must not hide the missing general SIMD
   and matrix realization family.
3. **Calibrated cost and search:** use MPP v2 and the CPU launch threshold as
   bootstrap priors, then measure emitted work, compile size, spills,
   cache/coherence, provider overhead and dispatch. Evaluate on disjoint
   shapes/operators; report held-out regret, top-K coverage and uncertainty.
4. **Production LLM coverage:** add hidden widths/context lengths, mask corner
   cases, dtypes and realistic prefill/decode sizes. Benchmark fused and
   unfused XIR/TIRx/Torch paths with identical inputs and explicit math policy.
5. **Generalize Metal MPP planning:** retain MPS, handwritten MPP, original
   TIRx, staged TIRx-MPP and native-MPP controls. Extend the legal realization
   family and test v2's rectangle/K/thread features on held-out GEMMs and
   production LLM operators. Do not turn the winning 128×32 schedule into a
   shape table; this cohort's GEMM parity is not universal library parity.
6. **Machine TileIR when needed:** promote realized maps, atoms, resource
   lifetimes and protocols into mutable typed records when multiple passes
   need them. Keep the public DSL minimal and avoid a new serialization layer.

Open questions are therefore concrete: which dependency-safe distribution
space pays off first, what calibrated features predict held-out performance,
and which physical realization explains the remaining library gap? The
current evidence supports pursuing those questions, not declaring completion.
