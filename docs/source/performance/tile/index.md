# Tile status and performance

```{toctree}
:hidden:
:maxdepth: 1

implementation
results
reductions
validation
checkpoints
```

## Current conclusion

As of September 7, 2026, on `codex/tile-programming-design`:
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

## How to read the performance evidence

Keep these objectives separate:

- **Batched E2E throughput:** warm host time per invocation, amortized over
  dispatches and synchronization; JIT/setup excluded.
- **Single-call E2E latency:** one dispatch through completion.
- **GPU timing:** instrumented compute-pass intervals plus a separate
  no-counter command-buffer control. The control includes GPU work and
  intra-buffer gaps, not isolated kernel time.

Counters perturb some Torch cases substantially. Every comparison retains
fusion, output-allocation and math-policy differences, paired round ratios
and negative results. Historical experiments are not one matched leaderboard.

## Results by route

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

**CPU: provider wins are not direct-XIR parity.**
[Proved CBLAS and Accelerate realizations](results.md#cpu-tirx-reference-gaps-and-proved-provider-realizations)
improve admitted TIRx families; CBLAS beats eager Torch on seven of eight
replayed shapes. Direct XIR/SIMD has a working mapping solver and
[packet-index codegen repair](results.md#simd-packet-index-proof-closes-a-codegen-disconnect),
but all six measured GEMMs still lose to Torch. General packed/vector
microkernels and Tile distribution remain missing. Attention, CNN/filter,
sort and Top-K PoCs establish correctness, not broad optimized performance.

## Validation and next milestone

The latest matrix-extension check completes a full build: **17/19 integration
invocations pass**, including 5,565 Metal matrix assertions. Two existing
Metal source-string suites still reject an unrelated user-owned barrier-flag
edit; the worktree is not all green. Default-off controls validate 56 complete
native/Torch outputs across Metal/CPU, with byte-identical Metal and only
bijective TBAA-label changes in CPU LLVM. See [validation](validation.md).

Next: preserve scalar-DAG reuse and estimate its live-state/instruction cost,
then compare legal fusion/materialization candidates through staged/JIT
selection on held-out graphs and shapes. Physical K/reuse, launch resource
limits, large-matrix scaling and direct XIR performance remain open.
A lower model score or a small cohort win is not completion.

## Detailed evidence

- [Implementation coverage](implementation.md): contracts, generality and acceptance.
- [Performance by compiler route](results.md): timings, controls and regressions.
- [Metal reduction measurements](reductions.md): mapping/resource experiments.
- [Correctness and failure investigations](validation.md): executed checks.
- [Implementation checkpoints](checkpoints.md): historical cohorts and artifacts.

The {download}`benchmark guide <../../../../scripts/benchmark/tile_torch/README.md>`
defines reproduction and timing modes. Exact commands and binary/source
fingerprints identify each experiment, not just its Git revision.
