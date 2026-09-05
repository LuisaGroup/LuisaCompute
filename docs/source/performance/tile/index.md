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

As of September 5, 2026, on the `codex/tile-programming-design` branch:
**the architecture runs, and several measured cohorts beat eager Torch or
approach/beat MPS, but the general performance goal is not complete.**
The results below are primarily FP32 on an Apple M1 Max. They do not establish
production-size LLM, low-precision, all-shape or cross-device parity.

The source language, mutable TileIR, C++ TIRx bridge, bounded native Metal MPP
lowering and XIR/SIMD CPU route are implemented. Execution mapping is planned
rather than mechanically copied from logical hierarchy. Cost policies are
backend-overridable, but their candidate families and calibration remain bounded.
See [implementation coverage](implementation.md) for exact limits and
[compiler architecture](../../internals/tile/index.md) for ownership and design.

## How to read the performance evidence

Three different timings are retained; none may be substituted for another:

- **Batched end-to-end throughput:** warm host time per invocation, amortized
  over dispatches and synchronization. JIT and setup are excluded.
- **Single-call end-to-end latency:** one dispatch through completion, including
  the Runtime/framework overhead.
- **GPU measurements:** instrumented Metal compute-pass timestamps and a
  separate no-counter command-buffer control. The control includes GPU work
  and gaps within the command buffer; it is not an isolated kernel timestamp.

The counter probe substantially perturbs some Torch cases, so it is diagnostic
rather than an uninstrumented speed ranking. Every external comparison must
also retain output-allocation, fusion and fast-math differences.

## Results by route

**Metal GEMM is near MPS, not decisively ahead of it.** In the saved 14-round
FP32 1024³ replay, TIRx-to-MPP views take 270.675 µs, native MPP 287.137 µs,
direct MPS 272.572 µs and eager Torch 284.654 µs in median host-wall batch time.
The paired TIRx-view/MPS time ratio is 0.9938: a small measured advantage.
Native MPP is not the winning route. These historical routes have different
recorded fast-math settings and are not a matched pure-kernel comparison.
The [route report](results.md) retains all shapes, controls and qualifications.

**Elementwise and row reductions have real structural improvements.** The
elementwise mapper now fuses logical-program and Tile-element coordinates,
and preserves shared pointwise SSA as worker-local values. Metal reductions
use SIMD-group collectives and compact worker-private stripes. The original
24-case row-program cohort beats eager Torch in host-wall throughput, with
Tile/Torch time ratios 0.032–0.902. Normalization/loss output allocation and
eager-versus-fused behavior qualify those ratios; they are not direct MPS
kernel speedups. See [route comparisons](results.md) and
[reduction measurements](reductions.md).

**Mapping and resource planning still have held-out failures.** A frozen
whole-launch cost profile improves the three 768×6144 norm/softmax cases,
but 37×1537 softmax and LayerNorm regress. Fixed-width/cache ablation then
separates mapping from input reuse. The latest worker-pack tail guard repair
improves 37×1537 Softmax/RMSNorm/LayerNorm batched E2E throughput by
1.134×/1.207×/1.210× against the previous emitter in four paired rounds.
GPU pairs for the two norms are mixed, and identical-source controls expose
background variability. No noisy labels were used to refit the model; the
new cost profile and input caching remain opt-in. These are separate
experiments, not gains that can be multiplied together.

**CPU provider wins do not close the native XIR gap.** Proved TIRx CBLAS GEMMs
beat eager Torch on seven of eight replayed shapes; Accelerate array operations
also improve the admitted reduction/softmax families. Direct XIR/SIMD has a
working execution-map solver and multi-operator correctness coverage, but lacks
a general high-performance matrix microkernel and Tile distribution family.
The [CPU route evidence](results.md) keeps provider and direct-XIR results separate.

## Validation and next milestone

The latest reduction implementation checkpoint completed a full build, 89
Python benchmark tests and 31 of 33 Tile CTests. CPU and Metal execution tests
pass, including 28 new tail/cache configurations. Two generated-source
assertions still conflict with an unrelated local barrier-flag edit; this is
not an all-green worktree. The width/cache experiment validates 120 outputs;
the frozen old/new tail replay validates 192. Their complete records are
linked from [reduction measurements](reductions.md).

The next milestone is stable, held-out mapping/resource selection after the
tail repair, with independently replayed acceptance and explicit GPU/E2E
objectives. Broader dtypes, layouts, production LLM shapes, native MPP coverage
and direct XIR performance remain open. A small cohort win is not the acceptance
criterion for "faster than MPS/Torch."

## Detailed evidence

- [Implementation coverage](implementation.md): supported behavior and acceptance work.
- [Performance by compiler route](results.md): external and native-to-native comparisons.
- [Metal reduction measurements](reductions.md): mapping, resource and codegen experiments.
- [Correctness and failure investigations](validation.md): executed tests and implementation fixes.
- [Implementation checkpoints](checkpoints.md): the historical sequence and original artifacts.

The {download}`benchmark guide <../../../../scripts/benchmark/tile_torch/README.md>`
defines reproducible runs and timing modes. Saved binary/source hashes and exact
commands, not just a Git revision, identify each performance experiment.
