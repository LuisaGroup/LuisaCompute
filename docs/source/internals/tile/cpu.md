# TIRx CPU execution and resource planning

These are LLVM/TIRx realizations of the shared structural export, not the separate [XIR/SIMD planner](xir.md). Each option retains its stated proof, resource and numerical boundary. [Runtime/provider integration](runtime.md#cpu-provider-realizations-through-tirx) and [CPU measurements](../../performance/tile/results.md#cpu-tirx-reference-gaps-and-proved-provider-realizations) have separate owners.

```{contents} On this page
:local:
:depth: 2
```

## LLVM compiler-temporary storage realization

`PlannerOptions::max_cpu_stack_bytes` is an opt-in storage budget (0–65536
bytes, default zero), independent of the logical execution hierarchy. It does
not change the C++ DSL, tile shape, loop order, arithmetic, input/output layout,
or CPU task binding. Disabling the planner retains the workspace path too.

~~~text
TileIR compiler temporaries / explicit Memory
                  |
     TIRx flatten + vectorize + unroll
                  |
  allocation identity + escape/placement audit
                  |
       compact local scalar storage?
       static positive extent, nonescaping?
       no manual/unknown allocation contract?
                  |
     cumulative aligned payload <= budget
          /                         \
        yes                          no
   LLVM stack allocation       TVM workspace allocation
   (may become registers)      (original mechanism)
~~~

The pass runs immediately before host builtin lowering, after transforms that
can expand storage or duplicate definitions. It charges each eligible static
allocation with 16-byte padding; branch copies are summed, not assumed to share
lifetimes. Vector-typed, dynamic, strided, custom-layout, aliased, raw-pointer,
address-taken, and manually materialized resources are not candidates. Explicit
Memory remains explicit even without a resource-class argument; its marker
survives the common passes until this audit. A 68-byte payload therefore needs
80 budget bytes; two such buffers need 160, not two independent 80-byte limits.
This bounds **newly planned temporary payload**, not the total thread stack:
LLVM spills, host wrappers, and user/TVM-owned stack objects are separate.

The pinned TVM host pass otherwise sends `local` allocations through
`TVMBackendAllocWorkspace`, even for small static tiles. This plan uses TVM's
native `disable_lower_builtin` allocation annotation to retain `AllocBuffer`
for LLVM; it neither rewrites generated LLVM nor replaces a kernel with BLAS.
The ordinary TIRx realization remains selectable with budget zero. Removing
these allocation calls is not by itself a CPU GEMM microkernel: operand packing,
multi-row register reuse, temporal accumulator residency, and task/cache
partitioning still need independent models and measurements.

The {download}`first CPU storage replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-stack-replay/notes.md>`
improves paired medians over the previous lowering but also has per-round
regressions. The {download}`direct Torch/BLAS follow-up <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-stack-system/notes.md>`
still shows an approximately 11× 1024³ gap. This is a legal realization
candidate, not evidence of a solved CPU mapping/cost model.

## Cartesian CPU register packs

`PlannerOptions::max_cpu_vector_lanes` bounds the logical scalar count in a
Cartesian pack: 16 (default), 32, 64, or 128. Non-default values require LLVM
and automatic vectorization. This is a compiler/code-size budget, **not** a
hardware SIMD width, measured register count, or a new DSL hierarchy. Disabling
the planner retains the existing single-row automatic pack.

For the two innermost independent axes, choose a power-of-two column width
`W <= 16` and row count `R`, with `R*W <= budget`. The full-pack coordinate map
is `(m_min + rm*R + r, n_min + cn*W + lane)`, where `0 <= r < R` and
`0 <= lane < W`. Column tails cover only complete rows; row tails cover all
columns. These domains are disjoint and cover the original rectangle. All
outer axes and each element's temporal recurrence are retained.

~~~text
single-row packing                   Cartesian packing (R=4, W=16)

for row                              for row_pack
  initialize C[row, vector]             initialize C[4 rows, vectors]
  for k                                for k
    load B[k, vector]                    load B[k, vector]  <-- shared SSA value
    update C[row, vector]                update row 0 vector
                                         update row 1 vector
                                         update row 2 vector
                                         update row 3 vector
~~~

This is unroll-and-jam of independent element instances, not tensorization.
Only sequences, rectangular serial loops with element-invariant bounds/steps,
and element stores are distributed. Allocations, definitions, control flow,
unknown annotations, and lane-dependent temporal bounds retain the single-row
fallback. The semantic independent-element contract permits the interchange;
an MMA annotation alone does not. Arithmetic expressions and serial K order are
unchanged, including when reassociation is forbidden.

Rows remain separate contiguous TIRx vectors inside the common temporal loop.
The first prototype instead flattened row/column coordinates into a single
64-lane vector; the pinned TIRx emitter expanded the irregular address vector
into scalar loads/stores and regressed the 512³ probe. That implementation was
removed. The current emission exposes shared B loads to ordinary LLVM CSE,
without rewriting LLVM text or calling a GEMM library. It is still an opt-in
candidate: code size, tail work, packing traffic, and register pressure must be
measured, independently of the stack-allocation budget.

The {download}`four-round fixed-geometry replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-cartesian-replay/notes.md>`
improves seven paired median ratios but regresses 32³; the default remains 16.
The {download}`six-order CPU/Torch/BLAS comparison <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-cartesian-system/notes.md>`
still shows about an 8× 1024³ gap. Larger packs alone are not a complete CPU
mapping model.

## CPU immutable-input expressions

`CompileOptions::forward_readonly_tile_loads` also admits an independent LLVM
candidate. It stays default-off and is separate from stack placement and
Cartesian packing. The existing MPP policy remains strict: a cooperative
memory-input atom requires an unconditionally valid address.

For CPU scalar/SIMD consumers, a padded snapshot can instead become the same
**lazy guarded expression** at its use site. If the original copy is
`T[i] = if_then_else(G(i), A[F(i)], fill(i))`, a later `T[j]` may become that
expression with `i := j`, only after all of these checks:

- The entire invocation satisfies `noalias`; the external source has no
  stores or escapes, and no unknown effect can invalidate that fact.
- Every memory read in the address, guard, and fill expression is immutable
  too. Read-only A alone does not make `A[index_buffer[i]]` a stable snapshot.
- The unique complete initialization dominates every use, each consumer
  index is proved in the temporary's domain, and the guard implies a valid
  source address. An unknown proof keeps materialization.
- The temporary is compiler-owned, compact FP32 storage with no explicit
  Memory annotation, alias, or observed storage identity.

~~~text
Tile load snapshot
        |
  effects + dominance + bounds + resource constraints
        |
        +-- unknown / mutable / manual ---> retain snapshot
        |
        +-- proved immutable
                  |
          +-------+--------------------+
          |                            |
     LLVM consumer                cooperative MPP input
     retain lazy guard/fill       require unconditional address
          |                            |
     SIMD + storage planning      MPP resource + geometry planning
~~~

This does not turn a padded Tile into an unguarded pointer view. Bounds and
fill semantics survive substitution; ordered MMA keeps the same K recurrence.
The proof still rejects unguarded memory-dependent consumer indices, including
some mathematically bounded expressions outside its fragment. A pure lazy
branch whose path condition proves every temporary bound can now forward a
guarded gather without dropping its source bounds/fill expression. Unknown
cases remain correct snapshots, not promises of complete indirect-access
optimization.

Legality is not profitability. Forwarding trades copy/workspace traffic for
direct global strides and repeated predicates; compact packing may still win
through cache reuse or vectorization. Candidate selection must measure that
tradeoff at fixed geometry, thread request, stack budget, and pack budget.

The {download}`four-round forwarding A/B <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-views-replay/notes.md>`
confirms staging-array removal and 1.45–1.76× paired median improvements for
five regular shapes. In that historical revision both ragged GEMMs kept the same snapshot code;
their timing differences are not evidence of forwarding's guard cost.
The {download}`six-order CPU/Torch/BLAS comparison <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-views-system/notes.md>`
measures 5.541 ms versus 0.982/0.989 ms at 1024³. The cost model must record
actual realization and fallback coverage, not merely the requested switch.

## Full-vector guard specialization

Ragged immutable views exposed two separate lowering problems. First, the
native arithmetic proof did not always recognize `G => G` when coordinates
contained mixed-radix division/remainder. The forwarding audit now also
accepts bounds that are structurally present as conjuncts of the original
guard. Association/order and extra masks do not matter; an OR arm is never
treated as an assumption. All existing immutability/dominance checks remain.

Second, **legally forwarded does not mean efficiently vectorized**. The
{download}`retained failed experiment <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-guard-plan/notes.md>`
eliminated A/B storage but emitted only scalar input loads and scalar FMAs
for both ragged benchmark shapes. A lazy per-lane bounds check had scalarized
the contraction. More aggressive copy elimination alone was much slower.

Automatic CPU packing now proposes one full-vector fast path. Let `p` be
the available enclosing coordinates, `l` a lane of a width-`W` pack, and `G`
the selected lane-dependent guard conjuncts. Its sufficient precondition is

`U(p) = AND(g(p, l) for g in G for l in [0, W))`.

~~~text
lazy guarded input expression
              |
    compute U for this pack
              |
       +------+------+
       |             |
     U=true        U=false
       |             |
  SIMD fast arm    original guarded arm
  selected g=true same padding and bounds
       |             |
       +------+------+
              |
     same ordered recurrence
~~~

The fast arm replaces only those established Boolean facts. Scalar guards,
including the K coordinate and row bounds, stay in the expression. The slow
arm is the original loop, not an unchecked pointer access. Full and tail
arms do not execute together, and no FP zero products or recurrence steps
are discarded. This is a transformation of independent element domains,
not an MMA-only rule or an additional DSL primitive.

The same proof now handles lane-dependent statement guards around stores.
This matters even for plain ragged elementwise kernels: a predicated scalar
store inside a vector loop can force LLVM to scalarize an otherwise contiguous
full pack. A statement guard is versioned only when it has no `else` effect and
every lane-dependent condition contributes a proved pack predicate. Scalar
conditions remain in both arms. The 17×257 add control consequently fell from
the earlier approximately 2.84 µs observation to about 0.42 µs without any
provider call or change to its bounds semantics.

The precondition uses **every lane**, not merely endpoints: masks can have
interior holes. Candidate packs have 4–16 lanes and at most eight distinct
guard leaves; they receive one binary version, not a decision tree per
predicate. Only pure integer conditions already evaluated unconditionally
for every lane are selected. Memory reads, dynamic divisors, variables
defined inside the pack, predicated-read address expressions, unknown
effects, nested lazy arms, and possibly empty inner loops cannot supply
speculated facts. Native SSA renewal and statement simplification run before
vector lowering. Explicit vector scopes and Metal/MPP selection are unchanged.

Versioning increases code size and may increase JIT time; it does not prove
profitability. The planner must compare actual snapshot, lazy scalarized, and
full-vector realizations at fixed geometry. The guarded-view and automatic
packing options remain independently opt-in. Tests check actual allocation
removal and vector FMA emission, plus nonzero padding, interior masks,
negative offsets, nonzero loop minima, dynamic/zero-trip recurrences, and
untaken expressions that would overflow or divide by zero if speculated.

The {download}`fixed-geometry, frozen-binary A/B <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-guards-replay/notes.md>`
measures 1.279×/1.818× paired median improvements for the two ragged GEMMs,
with all four pairs improving for each. The six regular shapes have unchanged
LLVM instructions and remain no-op timing controls. The independent
{download}`six-order library comparison <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-guards-system/notes.md>`
still measures 5.919 ms Tile versus 1.021/1.028 ms Torch/BLAS at 1024³; this
repair is not the end of CPU execution/target planning.

## CPU root launch-cost guard

Logical `parallel` still means independent instances. It does not require the
CPU target to pay a thread-pool launch for every tiny automatic root. The
current planner uses a transparent bootstrap rule:

~~~text
explicit worker scope                       -> parallel (hard constraint)
planner disabled                            -> parallel (reference policy)
automatic root, extent >= 64                -> parallel
automatic root, extent < 64, expensive body -> parallel
automatic root, extent < 64, cheap body     -> serial
~~~

The body audit recognizes transcendental TIRx operations by typed `Op`
identity. Unknown external and packed calls are conservatively expensive.
Known synchronous vDSP reductions are treated as cheap so several short rows
do not pay a host launch only to call a small array routine. This is a target
scheduling choice, not a proof of `parallel` independence and not a new DSL
scope. The value 64 is a replaceable prior; a calibrated model should use
task count, per-task work, provider overhead, thread-pool state and cache
footprint. Tests cover the boundary, a transcendental body, explicit worker
binding and the planner-disabled reference path.

## Shared Tile SSA, target materialization, and CPU provider atoms

Lazy Tile expressions are valuable for fusion, but structural export must not
erase a shared SSA boundary by cloning its producer. The default exporter now
preserves every pure multi-consumer Tile as one compiler-owned logical
materialization. This is independent of memory scope and creates a target
resource candidate, not a user-visible `Memory`. `EXPENSIVE_ONLY` remains an
explicit diagnostic/JIT alternative that preserves `exp`, `log`, `sqrt` and
`tanh` while allowing cheap arithmetic to fuse into consumers. Only the exact
shared FP32 `exp` expression currently carries a versioned provider contract,
and the target pass re-proves that expression rather than trusting the generic
materialization annotation.

The CPU provider proof is two-level:

1. TileIR structure proves semantic intent. A reduction must be one rank-one
   FP32 recurrence with exactly one extracted element, one typed add/max/min,
   one yielded carry, and the corresponding `0`, `-inf`, or `+inf` identity.
   Whole GEMM requires the complete proved `C=A*B` dataflow contract.
2. The target pass revalidates the transformed TIRx. Array math requires a
   static compact zero-based map or contiguous final-dimension recurrence;
   CBLAS requires three compact rank-two FP32 parameters with matching extents
   and `noalias`. A stale annotation cannot rescue a mismatching body.

```{figure} ../../../_static/tile/tirx-realization-pipeline.svg
:alt: Structural TileIR export creates versioned proof contracts; target-specific passes revalidate them before choosing portable, CPU-provider, or Metal matrix atoms.
:width: 100%

The second proof firewall lets common passes evolve without turning annotations
or diagnostic names into unchecked rewrite authority.
```

`CpuMatrixBackend::CBLAS` replaces the eligible whole function with one
registered `tvm.contrib.cblas.matmul` packed call. It does not decompose the
execution hierarchy mechanically or call CBLAS from ordinary elementwise
code. `CpuMathBackend::ACCELERATE` maps proved reductions to synchronous vDSP
and the shared exp map to vForce. Known wrapper pointers are nonescaping, so
the bounded compiler-temporary stack planner may still retain their compact
scratch storage.

The math option explicitly permits provider semantics: vDSP may reorder FP32
reductions, while vForce has documented denormal and exception differences.
Reference remains the default. On a build/target without the provider, an
explicit request fails. On a supported target, an unrecognized local pattern
stays in reference TIRx; it is never approximately matched by an opcode label.

The {download}`array-math replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-accelerate-ops-replay/notes.md>`
reports 2.71--6.12× paired gains for row sums and 2.10--5.46× for softmax,
while add controls remain approximately 1×. The
{download}`CBLAS replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-replay/notes.md>`
reports all eight shapes, direct CBLAS and eager Torch in all six execution
orders. The
{download}`CPU residual-LayerNorm search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-residual-layernorm-materialization-search/notes.md>`
selects recomputation for all four shapes while retaining structural SSA and
holding its native LLVM/input-view/vectorization/64 KiB stack policies fixed.
These results validate two reachable provider families and one target-specific
materialization choice; they do not imply that direct XIR or the portable
reference loops have reached library parity.
