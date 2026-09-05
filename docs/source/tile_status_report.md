# Tile programming: implementation and evidence report

As of September 5, 2026, Asia/Shanghai. This is a technical status report for
the `codex/tile-programming-design` branch and the source/report snapshot that
contains it. A Git revision alone does not reproduce performance experiments:
use the recorded binary/source hashes and exact commands in the linked
evidence.

## Technical summary

The packaged [XIR pilot report](../../scripts/benchmark/tile_torch/results/m1-max-20260905-xir-simd/report.html)
is an earlier bounded snapshot with charts, audit tables and canonical sources.
Its canonical validator and structural verifier pass. The packager found no
compatible Chromium headless-shell, so responsive/source-dialog browser QA is
not claimed; the self-contained semantic chart and table fallbacks remain in
the file. This Markdown page and its reading map are the current
repository-native report across all Tile routes.

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

The TIRx Metal route now has a third, non-library realization family for
proved FP32 row reductions. It maps one logical program to a target-legal
number of 32-lane SIMD groups, packs independent short programs, and compacts
eligible compiler-owned Tiles to worker-private stripes through an affine
ownership proof. Path-sensitive guarded-view analysis and an independent
distributed-local ownership audit now also make dynamic label gathers safe.
Across the current 24-case sum/softmax/RMSNorm/LayerNorm/cross-entropy/residual-
LayerNorm reports, every complete output passes and Tile/Torch host-wall
throughput ranges from 0.032× to 0.902×. Sum and softmax use preallocated
output on both sides; PyTorch's functional normalization/loss calls include
returned-output allocation. Same-binary four-round native A/B replays measure
21.19×--49.87× for RMSNorm, 14.04×--75.54× for LayerNorm/cross-entropy, and up
to 1.421× for preserving shared arithmetic in residual LayerNorm. A paired
CPU search selects the opposite recomputation policy, demonstrating that
logical SSA sharing and physical storage must remain separate decisions. This
is a narrow M1 Max cohort, not a production LLM suite or pure-GPU-event claim.

Automatic TIRx GPU elementwise mapping now fuses a logical program and its
independent Tile element coordinates into a physical thread grid. A four-round
frozen-binary add A/B validates all 32 outputs and measures 34.49×, 79.17×,
8.50× and 4.20× over the previous program-per-worker realization at 1×127,
17×257, 128×1024 and 4096×256. New times are 2.55, 2.74, 5.20 and 18.62 µs;
paired new/Torch time ratios are 0.704--0.832. This combines proved immutable
input forwarding with coordinate fusion; it is not just a launch-width tweak.

The same family now accepts bounded, pointwise shared Tile SSA chains and
turns each producer into one per-worker scalar definition. A same-binary
four-round GELU(A+B) A/B measures 32.70×, 52.64×, 9.91× and 3.83× over the
unfused program-per-worker map at those four shapes, with all 64 native/Torch
outputs valid. Both variants keep identical input-view and shared-SSA policy.
This is storage scalarization plus execution-grid fusion, not recomputation.
Its 2.59--19.93 µs times are still **host wall time**, and Torch is a two-op
eager graph with preallocated intermediate/output, not a compiled fused graph.

The benchmark now additionally supports **real Metal compute-pass timestamps
in a separate measurement phase**, alongside uninstrumented batched and
single-dispatch end-to-end times. A two-case integration smoke confirms that
GPU execution and dispatch latency differ substantially; it is not a stable
performance comparison. The M1 Max supports pass-boundary rather than arbitrary
per-dispatch counters, so multi-dispatch passes are labeled batch time. No
existing result has been relabeled as pure kernel time. The helper uses public
Metal APIs without changing the pinned TVMx build or Runtime interfaces.

The follow-up **audits the observer itself**: same-sized, alternating-order
samples also collect command-buffer GPU timestamps with no encoder probes or
counter attachments. Torch softmax's counter/control GPU time ratio reaches
3.08–5.85× in the diagnostic cohort. Accordingly, counter samples are retained
as diagnostics, not used as an uninstrumented kernel-speed ranking. Reports
now show the no-counter GPU control beside E2E timing for native/Torch/MPS.
This control includes all GPU work/gaps inside each command buffer, not just
one kernel. Six reduction cases, three SIMD-group/Torch/MPS GEMMs and three
native-MPP GEMMs pass complete output validation with the new control. Current
background-load variability rules out a stable ranking or cost-model update
from this cohort; prior host-wall A/B results keep their original scope.

Reduction collaboration width, independent-program packing and ordered stripe
unrolling are now separately controllable and jointly searchable through
ordinary staged/JIT runs. The TIRx cost model has a backend-overridable abstract
policy, while proofs and candidate legality remain bridge-owned. A separate
four-round, 80-output replay finds stable 1.107× for 1024×4096 sum, 1.062× for
1024×4096 softmax and 1.051× for 1024×257 softmax over the existing automatic
reduction mapper. The other seven shapes are flat/noisy or slightly worse;
the unchanged 17×257 sum is a noise control. Therefore the default unrolling
factor and coefficient prior remain unchanged. Sixteen additional norm/loss
outputs validate the new unrolled codegen, not a universal performance win.

The next structural reduction candidate is **consecutive elements per worker**:
`i=(chunk*workers+worker)*V+element`, V=1/2/4/8. Reducers, elementwise
consumers and compiler-local Tile storage share this ownership map; full packs
are separated from guarded tails. V defaults to one and is exposed to the
backend cost policy and staged/JIT search, not hardcoded for RMSNorm. An
explicit GPU-control JIT objective now complements the unchanged host-wall
default. Its score uses no-counter command-buffer GPU throughput, never the
perturbing compute-pass probe. The
{download}`implementation/validation checkpoint <../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-lane-validation/notes.md>`
records 24 new Metal layout cases, 82 Python tests and the full 31/33 Tile
regression: two existing source assertions conflict with the untouched local
`mem_flags(2)` edit. A separate four-round, 64-output frozen replay measures
**1.208× / 1.469× / 1.156× GPU-throughput gains** over the current V=1 mapper
for 1×127 / 17×257 / 64×4096 RMSNorm. At 64×4096 native GPU time is
9.216 µs versus Torch 9.172 µs: approximately parity, not a win. The
1024×4096 case is flat (1.015× GPU, 0.992× E2E throughput), and synchronized
single-call E2E latency still trails Torch at both wider sizes. Forty-eight
additional outputs validate six operators at V=4 without establishing a
cross-operator optimum. All figures keep no-counter GPU, instrumented probe
and E2E scopes distinct. Default V and coefficients remain unchanged pending
held-out mapping-model validation.

The
{download}`target-complete width checkpoint <../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>`
also repairs two omitted parts of the candidate space: TVM's benchmark target
inherited a 256-thread default, and the bridge restricted automatic
cooperation to powers of two through eight subgroups. The adapter now queries
the device limit and the solver includes every legal subgroup count up to the
32-partial collective bound. Fourteen new ragged softmax configurations pass,
including 96 and 1024 threads on this device. Physical group counts and useful
lane-work fractions reach backend cost policies and reports. Default scoring
coefficients are unchanged; the independent performance comparison is separate.

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
| [TIRx Metal reductions](tile_tirx_reduction_report.md) | Formal subgroup mapping, ownership proof, finite solver, staged/JIT interface, tests and complete performance evidence |
| [Runtime and native lowering](tile_native_runtime.md) | Factory, backend ownership, shader handles, Metal native/TIRx limits and ABI |
| [XIR execution planning](tile_xir_design.md) | Exact implemented CPU candidate space, cost equations, alias constraints, SSA/CFG lowering and extension roadmap |
| {download}`Benchmark tools <../../scripts/benchmark/tile_torch/README.md>` | Reproduction interfaces, timing definitions, controls and saved runs |
| {download}`Validation evidence <../../scripts/benchmark/tile_torch/results/m1-max-20260905-xir-validation/notes.md>` | Test commands, failure investigations, logs and audit details |
| {download}`Metal MPP cost v2 search <../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-search/notes.md>` | v1 failure, v2 equations, hard legality and in-cohort model regret |
| {download}`Metal MPP cost v2 replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-replay/notes.md>` | Frozen schedules, 784 full-output checks and balanced MPP/MPS/Torch evidence |
| {download}`Metal reduction cohort <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-reductions/notes.md>` | Sum, softmax and RMSNorm plans, 12 complete outputs, timings, hashes and exact command |
| {download}`Balanced RMSNorm A/B <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-rmsnorm-replay/notes.md>` | Same-binary reference/subgroup causality check across four shapes and four balanced rounds |
| {download}`Metal row-program extension <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions/notes.md>` | LayerNorm and cross-entropy plans, guarded-gather repair, eight complete outputs and API-level timings |
| {download}`Balanced row-program A/B <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions-replay/notes.md>` | Same-binary LayerNorm/cross-entropy causality check: 64 valid native variants and unchanged artifacts |
| {download}`Residual LayerNorm materialization search <../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-search/notes.md>` | Separate preserve/recompute JIT candidates, complete Metal outputs and exposed v1 cost-model regret |
| {download}`Residual LayerNorm materialization A/B <../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-replay/notes.md>` | Four balanced same-binary rounds isolating the shared-SSA decision |
| {download}`Bounded Metal thread search <../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-bounded-thread-search/notes.md>` | Exact width candidates, rejected over-budget stripes and fresh winner validation |
| {download}`CPU materialization search <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-residual-layernorm-materialization-search/notes.md>` | Opposite target choice under fixed LLVM/input-view/vectorization/stack policies |
| {download}`CPU CBLAS replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-replay/notes.md>` | Eight frozen GEMMs, six implementation orders, direct CBLAS overhead and Torch comparison |
| {download}`CPU array-math replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-accelerate-ops-replay/notes.md>` | Causal reference/Accelerate A/B over add controls, row reductions and softmax |
| {download}`Earlier CPU/provider validation <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-provider-validation/notes.md>` | Historical provider checkpoint, with its then-current build/test counts and documentation QA |
| {download}`Shared-Tile validation <../../scripts/benchmark/tile_torch/results/m1-max-20260905-shared-tile-validation/notes.md>` | Previous shared-SSA checkpoint, its 32/32 submitted-source Tile cohort and 69/69 benchmark contracts |
| {download}`Element-grid A/B <../../scripts/benchmark/tile_torch/results/m1-max-20260905-element-grid-replay/notes.md>` | Frozen old/new binaries, four balanced rounds, complete add outputs and the combined forwarding/fusion gain |
| {download}`Reduction joint-map replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-joint-map-replay/notes.md>` | Collaboration/packing/unrolling choices, three stable gains, seven qualified outcomes and raw per-round evidence |
| {download}`Execution-map validation <../../scripts/benchmark/tile_torch/results/m1-max-20260905-execution-map-validation/notes.md>` | Latest build, exact-constraint failures, CPU/Metal regressions, benchmark contracts and documented limitations |
| {download}`Shared element-SSA A/B <../../scripts/benchmark/tile_torch/results/m1-max-20260905-shared-element-replay/notes.md>` | Four balanced GELU(A+B) rounds, same-binary fusion isolation, complete outputs and eager-graph caveats |
| {download}`GPU/dispatch integration smoke <../../scripts/benchmark/tile_torch/results/m1-max-20260905-device-timing-smoke-v2/notes.md>` | Real pass-boundary GPU counters versus uninstrumented host dispatch; raw calibration and explicitly preliminary evidence |
| {download}`Dual-timing validation <../../scripts/benchmark/tile_torch/results/m1-max-20260905-dual-timing-validation/notes.md>` | Full 33-test Tile checkpoint, then-77 Python contracts, timestamp tests and submitted/worktree distinction |
| {download}`GPU timing observer audit <../../scripts/benchmark/tile_torch/results/m1-max-20260905-device-timing-counter-control/notes.md>` | No-counter GPU control, probe perturbation, reduction/GEMM/native-MPP integration and the preceding 80-test Python checkpoint |
| {download}`Consecutive-worker reduction layout <../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-lane-validation/notes.md>` | Generic ownership map, GPU-objective JIT, four-round RMSNorm GPU/E2E replay, six-operator coverage and current verification boundaries |
| {download}`Target-complete reduction widths <../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>` | Queried device limit, all whole-subgroup widths, immutable cost-policy features, 14 new GPU layouts and current 83-test Python checkpoint |

```{figure} ../_static/tile/xir-planning-pipeline.svg
:alt: The same execution-first TileIR feeds XIR/SIMD, Metal-native MPP and TIRx compiler routes with separate ownership.
:width: 100%

One language and Runtime contract do not imply identical target schedules or interchangeable benchmark results.
```

## Architecture decision ledger

This table is the compact answer to the design discussion. Detailed proofs,
syntax and implementation evidence live in the linked documents above.

| Question | Decision | Consequence |
|---|---|---|
| Programming model | Execution structure first, not an algorithm graph with a schedule attached afterwards | Lexical Nest structure exists before target mapping; operations inherit anchor/frontier from it |
| Core Nest vocabulary | `parallel`, `serial`, `pipeline`, `reduce` only | Elementwise, convolution, softmax, attention, Top-K and sort remain composable libraries unless irreducible target semantics justify an atom |
| C++ surface | Luisa-style staged lambda parameters, range-for Nests, ordinary carried assignment, explicit memory effects | No `GemmSpec`, public builder prefix, symbolic-integer façade, `loop.result()` or kernel-specific `mma_team` entity |
| Tensor and view | Tensor is storage plus a typed layout/view; `A[...]` loads a Tile, `A(...)` names a `MemoryRef` | Subtiles and bounds are explicit without baking execution or a memory hierarchy into Tensor |
| Layout algebra | One typed mixed-radix/index-map composition algebra for execution binding, value distribution, views, addresses and atom operands | Domain/codomain and proof obligations prevent composing unrelated coordinate spaces; representability is broader than any one emitter |
| Execution versus memory | Execution hierarchy chooses participants; resources attach independently to an owner prefix and access map | Several differently laid-out memories may serve one Nest; memory kinds are capabilities, not a fake total hierarchy |
| Pipeline | A temporal producer/consumer Nest with lexical stage cuts and dependence distances | It may organize participant specialization, overlap and versions, but a stage name alone does not promise async hardware |
| Reduction | An algebraic Nest with domain, grouping, identity/update/merge and numerical policy | Serial fold, subgroup tree, Welford and tuple states are realizations of one semantic contract, not unrelated source constructs |
| Tile SSA versus `Memory` | Preserve semantic sharing; plan retain/recompute/materialize per target. Manual `Memory` means stable addressable identity | Compiler stripes/registers/workspace do not leak into ordinary kernels; manual writes always use `.store()` |
| TileIR | Thin, typed, mutable SSA/region IR with managed intrusive ownership/use lists and analyses | It is transformable like XIR/LLVM, not a SPIR-V-style serialization schema and not an MLIR dependency |
| Backend boundary | Public `tile::compile(device, TileIR)` calls the optional backend `DeviceInterface::create_tile_kernel` factory, which selects native lowering or `tile/bridge/{tirx,xir}` | TIRx and XIR remain comparable bootstrap paths while Metal/CUDA/CPU keep target-specific bindings and atoms |
| Planning | Solve binding `B`, distribution `D`, atom `A`, resources `R` and schedule `Theta` under hard proofs, then rank | Enumeration/Pareto DP are implemented for bounded families; MILP, CP-SAT, beam or annealing are optional search engines, never legality oracles |
| Autotuning | Ordinary concrete host configurations are recaptured and JIT-compiled as a finite product | No capture-once super-kernel is required; every candidate and the fresh winner receive the full correctness oracle |
| Machine TileIR | Add it only when several backends/passes need a common scheduled atom/resource/protocol form | Current bridge-local plans remain honest stepping stones; no premature backend instruction serialization |

“Layout completeness” therefore has three separate meanings. The algebra is
closed over the admitted typed finite maps and can embed the CuTe-style
mixed-radix constructions used here; proof procedures intentionally return
unknown outside their decidable fragments; emitters support smaller target
subsets and fail closed. A complete representation never licenses an
unsupported lowering.

## What is implemented, and what remains design

| Area | Implemented and exercised | Important remaining boundary |
|---|---|---|
| C++ surface | Signature parameters; range-for Nests; direct carried assignment; explicit stores; Tile-level operations | Not arbitrary C++ capture or intra-kernel SIMT/Tile mixing |
| Execution | `parallel`, `serial`, `pipeline`, `reduce`; scope constraints | The backend must realize the requested binding; unsupported bindings are errors |
| Data/layout | Typed layout representation and proof mechanisms; Tensor as storage plus layout/view | Not every represented layout has an emitter on every bridge |
| TileIR | Mutable typed SSA, regions and intrusive ownership/use structure | General Machine TileIR and its pass suite are not implemented |
| TIRx | Native C++ export preserving pure multi-consumer SSA; target-selectable recomputation; CPU/Metal realizations; typed MPP v2 modes; proved Metal FP32 subgroup reductions; bounded target-specific cost/solvers | Materialization model lacks traffic/spill calibration; richer reduction policy and broader atoms/operators remain necessary |
| Native Metal | Typed FP32 MMA/view-forwarding subset; ordinary Runtime shader and launch | Not general epilogues, K pipelines, manual Memory, all dtypes or arbitrary operators |
| XIR/SIMD | Direct verified XIR; local Tile expansion; loop PHIs; ordinary CPU Runtime | No matrix-extension atom, packed GEMM microkernel or general Tile distribution |
| CPU planner / realizations | Root-axis permutations × legal worker-block widths; bounded storage/SIMD/launch choices; proved CBLAS and Accelerate atoms | Provider selection is explicit; no fitted break-even model, whole-program optimum, general Tile partitioning or physical pipeline solver |
| Autotuning | Recapture/JIT variants, Cartesian execution/resource/materialization candidates, exact Metal reduction-width sweeps and frozen-plan benchmarking | Broader search requires legal emitters and measured ranking; one capture is not mandatory |

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

The submitted source preserves `metal::mem_flags(3)`. The earlier
{download}`submitted-value checkpoint <../../scripts/benchmark/tile_torch/results/m1-max-20260905-dual-timing-validation/notes.md>`
passed **33/33** `test_tile_*` entries: 30 unit-labeled tests and three
integration tests. The latest target-complete-width follow-up rebuilt
the full selected tree and reran all 33 entries without touching the user's
pre-existing local `mem_flags(2)` edit: **31/33 passed**. The two failures
are generated-source assertions requiring `3` in
`test_tile_tirx_cooperative_metal` and `test_tile_tirx_memory_metal`; their
numerical checks pass. Neither assertion was weakened and the local edit is
not submitted. Current benchmark Python contracts pass **83/83**; the prior
24 ownership-layout cases and 14 new wider/non-power-of-two layouts also pass
in the execution test. The
{download}`current validation and full log <../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>`
keeps this dirty-worktree result separate from the earlier submitted-value
checkpoint. No whole-repository test pass is claimed.

## Five failure investigations changed the implementation

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

### Distributed initialization does not create shared private storage

The first cross-entropy subgroup run passed both reduction recognizers but
failed six of seven output rows. Generated Metal showed why: 256 workers each
had a private `float[4096]`, initialization wrote only the worker-owned stripe,
and thread zero later performed the dynamic label gather from its own mostly
uninitialized array. This is exactly the abstraction error that execution/
resource separation is meant to prevent.

The repair makes immutable-view analysis path-sensitive for guarded indirect
indices, producing a direct guarded Tensor read, and adds a separate
whole-program ownership audit for every distributed nonscalar local buffer.
An unknown ownership proof now declines the optimized mapping. A positive
cross-platform guarded-view test and a negative explicitly materialized Tile
test protect both decisions; the full explanation and diagram are in
[the reduction report](tile_tirx_reduction_report.md).

### Shared SSA must survive until target resource planning

Fused residual LayerNorm exposed a different structural failure. The old
exporter preserved shared transcendental expressions but cloned cheap shared
arithmetic into every consumer. `combined = X + residual` was consequently
expanded four times in generated Metal, multiplying global input reads even
though the later subgroup mapper had enough ownership information to retain a
compact value.

The exporter now preserves every pure multi-consumer Tile definition by
default. This is logical SSA, not a source-level `Memory` allocation. The
Metal mapper may materialize it as bounded worker stripes after an affine
ownership proof; a target may instead choose the explicit `EXPENSIVE_ONLY`
recomputation candidate. A 64-scalar-per-worker software-state bound rejects
pathological candidates before code generation. Metal selects preservation
for all four measured shapes; LLVM CPU selects recomputation for all four.
The [language/layout design](tile_programming_design.md) and
[formal reduction report](tile_tirx_reduction_report.md) record the full
contract and shared-Tile planning diagram.

## Performance: preserve the measurement basis

Unless explicitly labeled otherwise, reported times are **warm synchronized
host-wall time per invocation**, amortized over a batch. They include each
runtime's dispatch/encoding/submission and synchronization. They exclude JIT,
setup allocations/uploads and cold-call setup; returned-output allocation
stays inside timing where the recorded operator API requires it. They are not GPU hardware-event times,
and CPU thread requests are not measurements of actual library worker use.

Report tables use medians of within-round p50s. A paired ratio is the median
of same-round numerator/denominator ratios, **not** a ratio of the displayed
medians. Ranges and counts of slower rounds are descriptive, not confidence
intervals. No slow or failed row is discarded to improve the headline.

### Metal subgroup reductions close the measured normalization defect

The [formal design and evidence report](tile_tirx_reduction_report.md)
documents the new opt-in TIRx Metal realization. It structurally revalidates
canonical FP32 add/max/min reductions, searches whole-SIMD-group cooperating
widths, and derives private/shared storage from the selected execution
map. For softmax width 4096, a logical compiler-owned 4096-element Tile becomes
a compact private stripe (16 values at 256 threads) only after every access proves the same affine
owner; the old per-thread `float[4096]` form is rejected by source tests.

The two original current-binary cohorts use 11 samples, 100 ms calibrated
sample windows and 100 ms warmup. All 20 complete FP64 checks pass:

| Family | Shapes | Tile/Torch range | Fastest absolute Tile | Slowest relative Tile |
|---|---|---:|---:|---:|
| row sum | 1×127, 17×257, 128×1024, 64×4096 | 0.293×--0.716× | 3.106 µs | 0.716× |
| softmax | same widths/row counts | 0.124×--0.286× | 3.305 µs | 0.286× |
| RMSNorm | same widths/row counts | 0.546×--0.902× | 3.904 µs | 0.902× |
| LayerNorm | same widths/row counts | 0.511×--0.648× | 4.500 µs | 0.648× |
| cross-entropy | same widths/row counts | 0.032×--0.052× | 3.449 µs | 0.052× |

The four additional residual-LayerNorm cases search both shared-Tile policies
with separate capture/JIT/full validation, then recapture the winner. Metal
selects `PRESERVE` in every case:

| Rows×width | Tile µs | eager Torch MPS µs | Tile/Torch | Worker stripe scalars |
|---|---:|---:|---:|---:|
| 1×127 | 3.426 | 10.671 | 0.321× | 4 |
| 17×257 | 3.655 | 11.705 | 0.312× | 6 |
| 128×1024 | 6.321 | 18.592 | 0.340× | 8 |
| 64×4096 | 8.324 | 27.046 | 0.308× | 32 |

The independent same-binary replays rotate variant and case order for four
rounds. The subgroup path is 21.19×--49.87× faster for RMSNorm and
14.04×--75.54× for LayerNorm/cross-entropy by median paired ratio. Native uses
preallocated output; PyTorch's functional normalization/loss calls allocate
their returned output inside timing, so only the native reference/candidate
A/B is the clean causal comparison. The saved
{download}`cohort report <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-reductions/notes.md>`,
{download}`balanced replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-rmsnorm-replay/notes.md>`,
{download}`row extension <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions/notes.md>`
and
{download}`extension replay <../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions-replay/notes.md>`
and the
{download}`residual materialization search <../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-search/notes.md>`
retain every sample, plan, output error, artifact hash and exact command. The
separate
{download}`materialization A/B <../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-replay/notes.md>`
isolates the Metal decision: median paired preservation speedup is 1.057×,
1.008×, 1.354× and 1.421× from smallest to largest shape. The analytic v1
model does not count duplicated global loads or expression depth and incurs up
to 43.66% regret; the report preserves that miss rather than calling it a
model success.

This closes the diagnosed scalar-worker realization for the admitted subset.
It is not production attention, training-loss/backward coverage,
low-precision evidence, held-out device calibration or pure Metal kernel
timing. In particular, the very large cross-entropy advantage includes
PyTorch's general eager API and returned-output overhead; it is not presented
as an isolated MPS-kernel ratio.

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
GEMM or an exact reduction recurrence; structural export preserves every pure
multi-consumer Tile SSA by default. The CPU pass then revalidates the actual
TIRx body, buffer ABI, layout, alias contract and target policy before choosing
a resource or provider atom. It never matches a diagnostic operation name,
and an explicit unsupported request fails rather than silently changing
semantics.

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

#### Shared SSA and reduction realization

The structural exporter preserves a shared `exp` Tile once when its SSA result
has multiple consumers, instead of expanding the lazy expression into both a
reduction and an output consumer. The same default preserves cheap shared
arithmetic, but only a structurally revalidated `exp` contract can select the
provider below. The opt-in
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

#### Target-specific residual-LayerNorm materialization

The
{download}`CPU materialization search <../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-residual-layernorm-materialization-search/notes.md>`
holds native LLVM code generation, input-view forwarding, automatic element
packing, eight host threads and a 64 KiB compiler-local stack budget fixed.
Every one of its four measured winners uses `EXPENSIVE_ONLY`: 0.252, 8.799,
36.271 and 70.599 µs for widths 127, 257, 1024 and 4096. The corresponding
Tile/Torch ratios are 0.109×, 0.225×, 0.382× and 0.643×.

Metal selects `PRESERVE` on the identical semantic kernel because its mapped
worker stripes avoid repeated global reads. CPU benefits from recomputation
and LLVM fusion. This is direct evidence that preserving SSA in Candidate
TileIR does not imply a universal physical allocation; materialization belongs
beside binding, distribution and atom selection in the target plan.

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
3. **Calibrated cost and search:** use MPP v2, the CPU launch threshold and the
   exposed residual-LayerNorm regret as bootstrap evidence. Add duplicated
   global/local traffic, expression depth, live-state and measured spill
   features, then evaluate on disjoint shapes/operators; report held-out
   regret, top-K coverage and uncertainty.
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
