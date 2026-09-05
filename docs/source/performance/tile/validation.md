# Tile correctness and failure investigations

This record distinguishes executed correctness checks from performance claims.
See [current status](index.md) for the latest bounded conclusion.

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

## Metal reduction validation checkpoints

The latest cooperating-packing checkpoint completes a full build, all 89
Python benchmark tests and 31/33 Tile CTests. CPU/Metal execution and planner
tests pass, including 36 new numerical configurations and six typed raw-IR
admission/fence cases. Outputs are checked against independent FP64 formulas;
guard rows and unused output columns must retain sentinels after three
dispatches. The independent benchmark audit checks 240 executed output
validation records, frozen plans/sources and 21 unchanged binary artifacts.
The {download}`packing validation note and full CTest log
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cooperating-packing/notes.md>`
retain the same two unrelated source-assertion failures, not an all-green result.

### Earlier submitted-value and service-policy checkpoints

The submitted source preserves `metal::mem_flags(3)`. The earlier
{download}`submitted-value checkpoint <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-dual-timing-validation/notes.md>`
passed **33/33** `test_tile_*` entries: 30 unit-labeled tests and three
integration tests. The service-policy follow-up rebuilt
the full selected tree and reran all 33 entries without touching the user's
pre-existing local `mem_flags(2)` edit: **31/33 passed**. The two failures
are generated-source assertions requiring `3` in
`test_tile_tirx_cooperative_metal` and `test_tile_tirx_memory_metal`; their
numerical checks pass. Neither assertion was weakened and the local edit is
not submitted. At that checkpoint, benchmark Python contracts pass **89/89**; the planner
passes **5,988 assertions in ten tests**. The prior
24 ownership-layout cases, 14 wider/non-power-of-two layouts and 22 new
input-reuse numeric configurations also pass in the execution test. The
{download}`service-policy validation and full log <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-service-policy-validation/notes.md>`
keeps this dirty-worktree result separate from the earlier submitted-value
checkpoint. No whole-repository test pass is claimed.

### Earlier shared-Tile and resource checkpoints

The original shared-Tile run reported the following historical counts:

```text
complete CTest /^test_tile_/:            32 / 32 tests passed
guarded CPU view proof:               1,572 assertions passed
Metal subgroup LayerNorm:            12,297 assertions passed
Metal subgroup cross-entropy:            20 assertions passed
focused TIRx execution, CPU:          33,071 assertions passed
focused TIRx execution, Metal:        38,363 assertions passed
focused TIRx planner:                  5,891 assertions passed
Python benchmark contract discovery:    69 / 69 tests passed
```

That checkpoint temporarily tested the submitted memory-flag value and then
restored the local edit; the current run does not alter it. The
{download}`shared-Tile note <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-shared-tile-validation/notes.md>`
retains exact commands and the warning boundary. Do not relabel those counts
as a new clean-source run.

The subsequent target-width checkpoint adds 14 ragged Metal layouts and
passes 83 Python tests; input reuse adds 22 numerical configurations and
passes 84 Python tests. Access-demand validation then passes 87 Python tests,
89,942 focused input-reuse assertions and 5,941 planner assertions in nine
tests. Each rebuilds the selected tree and retains the 31/33 CTest boundary
without changing the local flag. The
{download}`target-width record <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>`,
[input-reuse evidence](reductions.md#budgeted-immutable-input-reuse) and
[joint mapping evidence](reductions.md#joint-resource-and-execution-mapping)
own their original logs. Later tail-pack validation adds 28 full/partial-pack
configurations with 89 Python contracts and the same CTest boundary; its
[fixed replay](reductions.md#tail-packs-a-structural-repair-after-width-reuse-ablation)
separately validates 192 benchmark outputs.

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
[the reduction report](../../internals/tile/reductions.md).

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
The [language/layout design](../../tile/design.md) and
[formal reduction report](../../internals/tile/reductions.md) record the full
contract and shared-Tile planning diagram.
