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

As of September 6, 2026, on the `codex/tile-programming-design` branch:
**the architecture runs, and several measured cohorts beat eager Torch or
approach/beat MPS, but the general performance goal is not complete.**
The results below are primarily FP32 on an Apple M1 Max. They do not establish
end-to-end production LLM, low-precision, all-shape or cross-device parity.

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

**Large Metal GEMM still falls short.** The earlier 14-round
[scale test](results.md#larger-matrices-the-1024-cubed-win-does-not-generalize)
covers 2048³/4096³/8192³, two large rectangles and a ragged shape. At 8192³,
native MPP's paired GPU/Torch time ratio is 1.985; TIRx→MPP views reaches
1.125 but loses all 14 GPU pairs. Two tail shapes reject the frozen large-view
schedule, while the other paths validate them. No new per-shape tuning is
claimed. There are 560 passing complete outputs and 28 retained rejections.

The subsequent [bounded-K view extension](results.md#bounded-k-mpp-views-legal-tails-remaining-library-gap)
admits three fixed K-tail requests, including 4096×4096×11008, without nominal
A/B shared staging. Its four-shape, seven-path replay passes 392 complete
outputs. Paired GPU view/MPS ratios for 1024×1024×1537, 4096×4096×11008 and
8192³ are 1.180/1.097/1.075; view/Torch ratios are 1.171/1.124/1.182.
Small-shape host throughput wins, but large shapes still lack parity and
retain substantial variation. The 8192³ view source is unchanged, so this
new session is not evidence of a compiler speedup at that shape. M/N-tail
forwarding and broader physical K/reuse choices remain open.

The subsequent [K/walk diagnostics](results.md#k-partition-and-program-walks-diagnostics-not-new-defaults)
find shape-dependent K sensitivity, reject simple row-stripe traversal, and
retain an inconclusive rectangle screen with order reversals. These are
exploratory benchmark results, not new production defaults or MPS/Torch wins.

The earlier result remains historical, not a scale guarantee. In its 14-round
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
new cost profile and input caching remain opt-in. Explicit packing now also
admits several cooperating programs per group, but the fixed 12-case replay
regresses in every GPU pair for eight cases and improves for only two.
Automatic packing is unchanged. These are separate experiments, not gains
that can be multiplied together.

The subsequent [fixed-total-group experiment](reductions.md#fixed-total-group-size-versus-automatic-execution)
also retains the automatic control: S2/P4 consistently improves two many-short-row
cases but regresses eight, despite its gains against some hand-picked mappings.
All 456 output validations pass; no new mapping default follows from this result.

The latest [wide-row coverage](reductions.md#wide-rows-and-large-working-sets)
reaches width 16384 and a 512 MiB input/output payload. Softmax, RMSNorm and
LayerNorm beat eager Torch in GPU and E2E throughput for all 18 cases in both
observed orders, with 72 full-output checks passing. These are only two rounds;
large RMSNorm margins narrow to about 1–4%, and single-call latency retains
mixed results and a regression. Throughput does not establish latency parity.

**CPU provider wins do not close the native XIR gap.** Proved TIRx CBLAS GEMMs
beat eager Torch on seven of eight replayed shapes; Accelerate array operations
also improve the admitted reduction/softmax families. Direct XIR/SIMD has a
working execution-map solver and multi-operator correctness coverage, but lacks
a general high-performance matrix microkernel and Tile distribution family.
The latest [packet-index proof](results.md#simd-packet-index-proof-closes-a-codegen-disconnect)
closes one codegen disconnect: four aligned GEMMs improve in every one of six
old/new throughput and latency pairs, with paired throughput time ratios
0.326–0.427. The ragged control retains identical LLVM and mixed results.
All six shapes still lose to Torch; this is not CPU parity or a new
multi-operator performance result.
The [CPU route evidence](results.md) keeps provider and direct-XIR results separate.

## Validation and next milestone

The latest reduction implementation checkpoint completed a full build, 89
Python benchmark tests and 31 of 33 Tile CTests. CPU and Metal execution tests
pass, including 36 new cooperating-packing configurations and six typed
admission/fence cases. Two generated-source
assertions still conflict with an unrelated local barrier-flag edit; this is
not an all-green worktree. The fixed packing experiment validates 240 outputs
across parameter pilots and the independent replay. Its complete records are
linked from [reduction measurements](reductions.md).

The new scale-benchmark orchestration passes 93 Python tests; it reuses those
same binaries and does not add a build/CTest claim. GEMM and reduction compiler
inventories are checked separately because the MPP path loads patched TVM.

The subsequent SIMD packet-proof checkpoint passes its four selected
Schedule/JIT/Tile Runtime/LLM CTests and all 95 Python benchmark tests. Its
final compiler replay validates 108 complete outputs, independently auditing
29,066,094 elements and 38 unchanged artifacts. It does not change the
unrelated worktree's broader CTest status. The new 8192³ MPS capture is saved
locally; Xcode inspection timed out, leaving large-shape counter attribution
open rather than inferred from the old 1024³ profile.

The bounded-K checkpoint adds 1,857 passing Metal matrix assertions in 28
tests, old-v2 compatibility, five selected CTests and 95 passing Python tests.
Its independent audit checks 392 full-output receipts, 26 unchanged artifacts
and eight deliberately corrupted evidence cases. This does not change the
unrelated barrier-assertion boundary described above.

The next milestone is M/N-edge matrix realization, physical K/reuse choices and mapping/resource
selection that scales beyond 1024³, with independently replayed acceptance and
explicit GPU/E2E objectives. Broader dtypes, layouts, production LLM workloads, native MPP coverage
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
