# Tile status and performance

```{toctree}
:hidden:
:maxdepth: 1

implementation
results
reductions
validation
checkpoints
migration
```

## Current conclusion

As of September 9, 2026, development is on `next` (merged at `2d02721b8`):
**the architecture runs, but the general MPS/Torch performance goal is not
complete.** Several bounded FP32 cohorts on Apple M1 Max beat eager Torch;
large GEMM and direct XIR/SIMD still have substantial gaps. These results do
not establish production LLM, low-precision, all-shape or cross-device parity.

The C++ language, mutable TileIR, native C++ TIRx bridge, bounded Metal MPP
lowering and XIR/SIMD Runtime route are implemented. Execution mapping is
planned rather than mechanically copied from logical hierarchy. Backend-owned
cost policies are extensible; their legal candidates and calibration remain
bounded. See [coverage](implementation.md) and
[compiler architecture](../../internals/tile/index.md).

The [legacy migration / low-precision checkpoint](migration.md) adds reusable
examples and FP16/BF16 coverage. Its diagnostic comparisons retain unstable
FP16 wins/regressions and large scan regressions; they do not replace the qualified
Torch/MPS scoreboard below.

## How to read the performance evidence

Keep these objectives separate:

- **Batched E2E throughput:** warm host time per invocation, amortized over
  dispatches and synchronization; JIT/setup excluded.
- **Single-call E2E latency:** one dispatch through completion.
- **CPU native entry:** actual emitted kernel entry called from C++, with
  Runtime dispatch/Python/JIT and caller allocations excluded. Required native
  traversal, launch resets and compiler-emitted internal allocations stay
  inside; this is not a hardware cycle counter.
- **GPU timing:** instrumented compute-pass intervals plus a separate
  no-counter command-buffer control. The control includes GPU work and
  intra-buffer gaps, not isolated kernel time.

Counters perturb some Torch cases substantially. Every comparison retains
fusion, output-allocation and math-policy differences, paired round ratios
and negative results. Historical experiments are not one matched leaderboard.

## Results by route

**Direct SIMD: three native operator families win; three still lose.**
The latest [24-case native comparison](results.md#native-row-entries-expose-both-broader-wins-and-remaining-gaps)
uses actual ORC and TorchInductor 2.14.0 entries, FP32 and one CPU thread,
at four sizes through approximately 4.2 million elements. Fixed local=8
RMSNorm, LayerNorm and GELU+residual respectively take 0.337–0.469×,
0.374–0.735× and 0.568–0.645× Inductor time; all 72 paired rounds win.
Masked softmax, SwiGLU and RoPE remain 1.350–1.607×, 1.166–1.244× and
1.247–1.944×; all their 72 paired rounds lose. Actual variance/reduction/math
implementation differences remain documented alongside full FP64 checks.

This new measurement uses the existing opt-in
[use-site memory realization](results.md#use-site-private-indices-unlock-contiguous-simd-memory),
not a new compiler optimization or calibrated automatic policy. The separate
440-visit E2E screen improves many large/local row programs but retains small
task regressions and non-winning attention. Shared-input/multi-output DAG
fusion now has a [guarded, opt-in implementation](../../internals/tile/xir.md#guarded-pointwise-dag-fusion-keeps-an-alias-safe-fallback)
with alias and bounds regressions; it is **not yet a measured performance
win**. A separate [expression/reduction fusion checkpoint](validation.md#expression-producers-join-their-first-reduction-traversal)
now merges two producer traversals in masked softmax and one in LayerNorm,
with full output checks at four sizes. It remains opt-in: an interrupted,
contended timing cohort is excluded, so the native ratios above are not
replaced by a new speedup claim.
The subsequent [next integration check](validation.md#next-integration-keeps-performance-qualification-separate)
keeps two contended native replay attempts out of the performance evidence.
Two later [native codegen probes](results.md#late-native-codegen-probes-separate-address-demand-from-inlining)
isolate scalar address demand and packet-loop inlining. Their correctness and
code evidence are complete, but both timing cohorts remain diagnostic-only;
the experimental overlays are not promoted into the working compiler.
[Independent task grain](results.md#cpu-task-grain-is-independent-of-the-native-packet-body)
and realization-sensitive cost calibration remain necessary; native superiority
alone does not establish dispatch latency or default-path parity.

**Metal matrix programs: broader legal lowering, incomplete profitability.**
Bounded K/M/N views and [direct output](results.md#bounded-output-removes-shared-c-not-the-whole-library-gap)
remove unnecessary staging using access/recurrence proofs. The
[realized-work model](results.md#realization-derived-work-and-model-selection)
improves three changed choices by 3.88–17.75% median GPU time, but all eight
GPU medians still lose to Torch. [Scale coverage](results.md#larger-matrices-the-1024-cubed-win-does-not-generalize)
extends through 8192³ and large rectangles; the historical 1024³ MPS
near-parity result does not generalize to those shapes.
[Traversal/participation experiments](results.md#generic-traversal-composes-with-k-but-is-not-a-universal-win)
retain regressions and have not produced a new universal default.

The latest [closed scalar epilogue](results.md#closed-matrix-epilogues-general-legality-mixed-profitability)
candidate uses ordinary expression DAGs, not activation-name rules. At fixed
256-worker schedules, ragged ReLU/GELU GPU medians improve 15.54%/14.41%,
but 4096³ GELU regresses 35.14%. All 288 complete native/Torch outputs pass.
The candidate stays **off by default**; removing storage is not a
profitability proof. Native-MPP and CPU/SIMD emission are unchanged.

**Metal elementwise and row programs: measured structural wins, with limits.**
[Multi-output pointwise fusion](results.md#multi-output-pointwise-fusion-removes-a-mapping-boundary)
uses common ownership/effect analysis through 4096×4096. Eight GPU medians
beat preallocated eager Torch (new/Torch 0.309–0.952), with mixed single-call
latency and one small-GELU GPU reversal. [Wide-row reductions](reductions.md#wide-rows-and-large-working-sets)
use SIMD-group collectives and compact worker-private stripes. Softmax,
RMSNorm and LayerNorm beat eager Torch in all 18 GPU/E2E throughput cases
in two observed orders, through width 16384 and 512 MiB payloads. Some
large margins are only 1–4%; latency and returned-output allocation differ.
[Cooperating-row packing](reductions.md#fixed-total-group-size-versus-automatic-execution)
and held-out cost policies still regress on several cases; they remain opt-in.

[Partitioned-output fusion](results.md#partitioned-outputs-remove-the-rope-mapping-fallback)
now handles proved disjoint regions of one buffer. Four RoPE GPU batch medians
are 0.129–0.319× preallocated eager Torch through 4096×4096, with all 24 paired
GPU comparisons winning. SwiGLU's unchanged single-output code measures
0.486–0.593× Torch; tiny single-call E2E still loses. These are bounded FP32
fusion results, not attention or direct-SIMD parity.

**CPU: provider wins are not direct-XIR parity.**
[Proved CBLAS and Accelerate realizations](results.md#cpu-tirx-reference-gaps-and-proved-provider-realizations)
improve admitted TIRx families; CBLAS beats eager Torch on seven of eight
replayed shapes. Direct XIR/SIMD has a working mapping solver and
[packet-index codegen repair](results.md#simd-packet-index-proof-closes-a-codegen-disconnect),
but all six measured GEMMs still lose to Torch. The
[private-vector realization](results.md#simd-local-distribution-and-private-layout-are-separate-decisions)
keeps mapping/layout fixed and realizes common-slot private accesses as
contiguous vectors. Native RMSNorm drops from 28.623 to 11.511 µs at 64×256
and from 7380.091 to 3336.237 µs at 1024×4096; paired new/Inductor time ratios
still remain 1.260/1.503. This is generic backend lowering, not a calibrated
new solver. The subsequent first-consumer load/reduction fusion is legal but
**default-disabled**: actual native RMSNorm regressions are 17–23% despite
lower estimated memory work. This points to missing full-packet
specialization/inlining and phase-cost interactions.
The earlier [full-packet specialization](results.md#full-packet-specialization-changes-the-profitable-local-mapping)
addresses one such interaction: at fixed packet-local mapping with fusion off,
native RMSNorm reaches 5.334 µs / 1117.151 µs for 64×256 / 1024×4096,
with paired new/Inductor ratios 0.603 / 0.511. All 12 corresponding paired
rounds win. This generic backend candidate remains opt-in, as does joint
mapping search; it does not establish default-path or all-operator parity.
The accompanying 240-visit Runtime E2E screen also improves broad norm and
GELU cases, while retaining narrow-row mapping regressions and weak SwiGLU
results. Native and E2E timing remain separate.
General packed
microkernels and arbitrary Tile redistribution remain missing. Attention, CNN/filter,
sort and Top-K PoCs establish correctness, not broad optimized performance.

The [Torch CPU inspection](results.md#torch-cpu-code-inspection-exposes-missing-local-vector-candidates)
has led to [indexable snapshots](results.md#indexable-xir-snapshots-remove-quadratic-extraction-work)
and [bounded traversal/private workspace](results.md#bounded-xir-traversal-improves-compilation-not-yet-torch-parity).
Those earlier checkpoints improve compilation/code size and complete all 11
large-shape cases, without establishing Torch parity. Local-vector distribution
now exists as a bounded opt-in candidate; safe snapshot/phase fusion and
independent CPU task grain remain open. The earlier private-vector 120-visit fixed-mapping
E2E screen also improves LayerNorm, softmax, SwiGLU and GELU, with non-winning
RoPE and narrow-local cases retained. Native-entry and Runtime E2E measurements
are reported separately rather than combined into a historical leaderboard.

The [LLM negative-result screen](results.md#attention-and-direct-simd-still-need-richer-execution-mappings)
measures six direct-SIMD operators at 1.22–16.35× Torch E2E time. The subsequent
[automatic cooperative mapping](results.md#automatic-cooperation-removes-the-attention-worker-fallback)
removes Metal attention's whole-program worker fallback: prefill/decode GPU
medians are now 80.612/327.875 µs, but still 2.894×/10.132× Torch SDPA in a
six-order replay. All 36 outputs pass. Scalar reductions, the second contraction
and shared-state traffic remain important structural work; SIMD is unchanged.
This broadens a generic mapping family, not a complete calibrated solver or
LLM performance parity.

The September 8 [composed-reduction follow-up](results.md#composed-reductions-need-phase-specific-contraction-distributions)
isolates input storage and group width: changing only the two closed
collectives gives small descriptive gains. A QK contribution-axis **benchmark
probe**, not a production planner change, regresses at 64 threads but improves
at 1024; three 1024-thread decode cases still take 3.80–9.64× Torch GPU time.
The study retains its non-interleaved configuration order and one rejected
Torch timing row. It motivates phase-specific partition candidates and joint
resource/transition costs, not an unconditional subgroup default for `mma`.

## Validation and next milestone

The latest [reduction-policy checkpoint](validation.md#per-operation-reduction-policy-checkpoint)
completes an independent full build, **35/35 Tile CTests** and 110 Python
benchmark tests. It preserves strict folds through final machine compilation
and admits closed subgroup reductions in composed programs. The original
worktree's separate run remains **34/35** because of an unfinished matrix
experiment excluded from this checkpoint. These are correctness results,
not new performance measurements. The earlier matrix default-off controls validate 56 complete native/Torch outputs
across Metal/CPU, with byte-identical Metal and only bijective TBAA-label changes
in CPU LLVM. See [validation](validation.md).

Next: distribute general Tile elements on XIR/SIMD and optimize reductions and
accumulator expressions within Metal's newly admitted composed programs.
Preserve scalar-DAG reuse and estimate its live-state/instruction cost,
then compare legal fusion/materialization candidates through staged/JIT
selection on held-out graphs and shapes. Physical K/reuse, launch resource
limits, large-matrix scaling and direct XIR performance remain open.
A lower model score or a small cohort win is not completion.

The CUDA reference PTX route (see [Runtime integration](../../internals/tile/runtime.md))
is implemented but not yet validated on an NVIDIA machine with a TIRx-enabled
TVMx build; its CMake TIRx gate must be on. Host artifact tests assert the
emitted CUDA source/PTX invariants and fail-closed options; `test_tile_cuda_ptx`
runs elementwise, reduction, softmax and GEMM oracle checks (transposes,
`allow_reassociation`, extrema) when built with the bridge and
verifies the explicit no-bridge diagnostic otherwise. No CUDA performance
numbers are claimed yet.

The [execution calculus](../../internals/tile/calculus.md) now separates primitive
assumptions, order, reduction laws and typed mapping obligations. Its sibling
fusion, joint resource/time solving and distributed plans remain proposals.
The [literature survey](../../internals/tile/related-work.md) identifies close
precedents; neither completeness nor research novelty is established.

## Detailed evidence

- [Implementation coverage](implementation.md): contracts, generality and acceptance.
- [Performance by compiler route](results.md): timings, controls and regressions.
- [Metal reduction measurements](reductions.md): mapping/resource experiments.
- [Correctness and failure investigations](validation.md): executed checks.
- [Implementation checkpoints](checkpoints.md): historical cohorts and artifacts.

The {download}`benchmark guide <../../../../scripts/benchmark/tile_torch/README.md>`
defines reproduction and timing modes. Exact commands and binary/source
fingerprints identify each experiment, not just its Git revision.
