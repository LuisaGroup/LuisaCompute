# TileIR → XIR: execution planning and SIMD realization

Status: executable CPU realization with bounded packet-index proofs and
compiler-owned snapshots, bounded Tile traversal, closed unordered partials and
an opt-in packet-local mapping, September 8, 2026. The finite solver below is implemented. General Tile distribution, packed
matrix atoms, software pipelining and measured cost calibration are not.

This document complements the [language/layout design](../../tile/design.md),
[target-independent planner formulation](planner.md) and
[Runtime integration](runtime.md). It does not redefine the DSL for
the CPU backend.

## 1. Preserve the execution-first program

The frontend declares **which independent programs and ordered computations
exist**, not a fixed CPU vector width or GPU block arrangement. For example:

```cpp
auto definition = tile_kernel("gemm", [=](TensorView<const float, 2> A,
                                         TensorView<const float, 2> B,
                                         TensorView<float, 2> C) {
    auto gm = axis("gm", ceil_div(M, BM));
    auto gn = axis("gn", ceil_div(N, BN));
    auto m = axis("m", BM), n = axis("n", BN), k = axis("k", BK);
    for (auto &nest : parallel(shape(gm, gn))) {
        auto m0 = nest.index(gm) * BM, n0 = nest.index(gn) * BN;
        auto acc = zeros<float>(shape(m, n));
        for (auto &step : nest.pipeline(shape(ceil_div(K, BK)),
                                        {.stages = 2, .initiation_interval = 1})) {
            step.stage("load");
            auto k0 = step.index() * BK;
            auto a = A.tile(coord(m0, k0), shape(m, k)).load();
            auto b = B.tile(coord(k0, n0), shape(k, n)).load();
            step.stage("compute");
            acc = mma(a, b, acc);
        }
        C(coord(m0, n0), shape(m, n)).store(acc);
    }
});
```

The loop variables are Nests, loads produce Tile SSA, assignment captures
loop-carried dataflow, and stores are explicit effects. No CPU-only `lane`,
`mma_team`, memory-owner annotations or builder-qualified math leaks into the
program. Several independent resources may be used by the same Nest.

## 2. Module and ownership boundaries

```{figure} ../../../_static/tile/xir-planning-pipeline.svg
:alt: One TileIR program feeds the planned XIR/SIMD path and separate Metal-native and TIRx routes.
:width: 100%

Planner, bridge, backend and Runtime are separate owners. No Python source or AST reconstruction is inserted between TileIR and XIR.
```

| Component | Owns | Does not own |
|---|---|---|
| TileIR | Typed semantic operations, nested regions, mutable use-def structure | LLVM, TVM, device queues |
| XIR planner | Finite candidate enumeration, target facts, cost decomposition | Runtime allocation or JIT |
| XIR lowerer | An owned XIR Module and typed argument/dispatch metadata | AST reconstruction or serialization |
| SIMD backend | Schedule/LLVM compilation, native shader and CPU dispatch | A second Tile language |
| Runtime adapter | Shader lifetime, argument/range checking, normal dispatch commands | Hardware scheduling policy |

Public headers live in `include/luisa/tile/bridge/xir/`; implementations live
in `src/tile/bridge/xir/`. The bridge links TileIR and XIR, not TVM, LLVM,
SIMD or Runtime. Native Metal lowering remains in the Metal backend. A future
backend may consume the same XIR result without relocating this bridge.

The input Module is borrowed and unchanged. The lowerer returns an owning
Module, not a dangling function pointer. Passes may rewrite its basic blocks,
PHIs, use lists and instructions; this is deliberately not a wire format.
The SIMD adapter runs the existing shared SSA optimization factory, then
CFG simplification and reachable-block verification. It does not rerun AST
destructuring or inlining on already plain SSA. Diagnostic LLVM capture is
independent of assembly capture, so normal compilation does not perform a
second machine-code compilation merely to retain source identity.

## 3. Formal mapping: ancestry plus local access

Let the root independent domain be a finite box
`D = [0,d0) × ... × [0,dr−1)`. A candidate permutation `π` is ordered from
outermost to innermost execution axis. Define:

```text
jπ(c) = Σt c[π(t)] × Πu>t d[π(u)]
c = unflattenπ(j),  0 ≤ j < P,  P = Πi di

block   = floor(j / B)
packet  = floor((j mod B) / W)
lane    = j mod W                 (B is divisible by W)
```

`B` is the logical workers per Runtime block; `W` is the backend's packet
width. Incomplete final packets are masked by the existing SIMD ABI. The
worker pool assigns blocks dynamically; a block is not pinned to a specific
OS thread. The permutation is a bijection of the **existing** parallel
instances. It does not require proving the independence already promised by
`parallel`, and it does not create new independent instances.

For a local Tile coordinate `e`, lexical descendant coordinates `s` and a
view origin `o`, the logical access is:

```text
ancestor coordinates c, descendant coordinates s, local Tile coordinate e
                        │
                        ▼
            logical buffer coordinate v = o(c, s) + e
                        │
                        ▼
      compact row-major element address = Σi v[i] × stride[i]
                        │
                        ▼
             bound BufferView byte offset + sizeof(T) × address
```

The execution map changes `j ↔ c`, not the buffer strides. Each buffer may
have a different origin, logical shape and access relation at the same scope.
This is the concrete subset of the general typed composition
`AddressMap ∘ ViewMap ∘ LocalAccess ∘ AncestorProjection`.

The current exporter materializes compact static buffer indexing only. The
language's richer layout algebra and proof system are not all realizable by
this exporter yet. Representability in TileIR must not be confused with
backend support. Unknown layouts/bindings fail closed.

Before creating bounds diamonds, the exporter derives integer intervals from
actual Nest coordinates and supported signed-i64 expressions. Checked
add/subtract/multiply and positive constant division/modulo may prove an
axis access in bounds. Negative offsets, unknown expressions and any possible
signed overflow keep the guard. This proof is separate from the cost model's
floating-point address-slope estimate. It neither asserts noalias nor moves a
load across an effect. LLVM simplification alone is too late to prevent
unnecessary per-element branches from inflating the earlier SIMD Schedule.

### Indexable snapshots preserve Tile value semantics

A Tile is still an immutable SSA value, not a deferred buffer view. The XIR
lowerer distinguishes access forms without changing the DSL:

- A constant coordinate, including an expanded Tile-map coordinate and
  supported checked integer expressions, directly selects the scalar SSA
  element. It does not build a full SELECT chain for LLVM to simplify later.
- A Tile with runtime-indexed extract users receives a compiler-owned local
  array at its definition. The lowerer stores its elements once, then emits
  a guarded GEP/load at each dynamic extract. Singleton/empty values and
  unproved statically expanded expressions retain the SSA fallback.

```text
buffer.load at definition
          │
          ▼
       Tile SSA
          │ save once
          ▼
   private snapshot[D]
          │ guarded indexed read
          ▼
    extract(index(r))

Later buffer.store changes the
buffer, not this snapshot.
```

For D input elements and R runtime extracts, the selected representation
replaces D×R selection work with D definition-time stores plus R indexed
reads. The eager baseline stores at definition; the admitted first-consumer
fusion below may instead initialize the snapshot during its first traversal.
Small loop carries use simultaneous PHIs; the bounded large-carry representation
below uses a staged parallel copy and preserves zero-trip initial state.
Distinct SSA definitions have distinct storage, preserving multi-Tile swaps.
An input parameter being `const` is not a noalias assertion: an overlapping
writable argument may overwrite the buffer without changing the snapshot.

This is a physical representation repair, **not contribution-axis vectorization**.
The SIMD emitter allocates each worker's local array separately within a
packet and uses its existing gather/scatter machinery. It may still spill.
`max_local_bytes` bounds
the sum of snapshot allocations per logical worker (default 256 KiB), not
peak liveness or the complete target stack; the packet multiplies this storage
by W. The existing SSA expansion budget also charges allocation/GEP/store
construction. Exceeding either bound rejects lowering rather than truncating
values or silently changing semantics. This does not introduce manual Memory
requirements or a new execution scope.

The guarded load retains this bridge's existing flat-index zero fallback;
it is not a new language-wide promise about invalid multidimensional
coordinates. Neither extraction representation changes fold L/R order,
reducer operand order, floating-point policy or the root execution mapping.

### Bounded traversal, partial reductions and private resources

The representation layer now has three forms. This is compiler state, not
three new user-visible Tile or Memory types:

| Value | Physical form | Evaluated at |
|---|---|---|
| Small Tile | SSA / indexed snapshot | Definition |
| Large shared value | Array + loop | Definition |
| Constant / single-use math | Splat / pure recipe | Consumer |

Large loads, maps and multi-consumer expressions normally use the array form.
External reads remain eager unless the first-consumer fusion below proves
that delaying them crosses no mutation or stage boundary. A deferred recipe
captures immutable physical operand definitions, not mutable entries in the
lowerer's value lookup table.

`max_unrolled_tile_elements` defaults to 64; zero explicitly selects the old
fully expanded diagnostic form without removing IR/storage budgets. Loop
instructions no longer grow with a large Tile's element count. This does not
bound total code size independently of the number of operations or nested
small expansions. Multi-consumer expressions remain materialized; there is
no calibrated recomputation/materialization search yet.

```text
source .load() ── bounded loop ──> immutable snapshot
                                       │
                             pure expression recipe
                                       │
                              bounded consumer loop

large carried state:
  init → current ── body ──> yielded values
            ▲                   │
            └── copy all ── next buffers
                           ▲
               stage every incoming value first
```

Large loop carries have current/next storage. All yielded values are staged
before any current value is overwritten, including cross-carry swaps.
Small scalar/Tile carries remain PHIs. MMA's contraction uses an ordered
runtime fold for a large contraction domain; this is not a packed matrix atom.

For a closed scalar `unordered_tree` reduction, the lowerer recognizes a
single ADD/MUL/MIN/MAX update whose carry has no other users and whose
contributions contain only constants, elementwise math and Tile extraction.
It splits contributions over `reduction_partitions` independent accumulators
(default 4, legal range 1–16). Each nonempty partition starts from an actual
contribution; the source initial value enters the final merge **once**, with
no invented zero/one identity. Tails are exact. Strict fold L/R, non-closed
state recurrence, multiple carries, nested regions and effectful bodies retain
their ordered fallback. This uses the unordered numerical contract, not a
claim that floating-point addition is exactly associative. These partials
remain inside one logical worker in the default complete-program mapping.
The opt-in mapping below also merges them across packet lanes.

### First-consumer fusion preserves the snapshot contract

The **opt-in**, default-disabled `enable_load_reduction_fusion` selects a
shared realization rule in the planner and lowerer. A materialized view load can join the first actual
consumer reached through single-use pure Tile expressions when that consumer
is a closed, scalar unordered reduction using bounded partials or packet
distribution. Unit-size map wrappers, including library `reduce()`, execute
once and are transparent. Other enclosing execution scopes are not.

The admission rule requires a bijection of matching nonunit dimensions and
extents and direct reduction coordinates in each extract. Unit axes may be
inserted/projected; a complete multidimensional reduction may permute axes.
It rejects reordered element indices, strict folds, unknown effects, every
intervening write (even through a different buffer argument), and `stage`
boundaries. This does not re-prove the independence promised by `parallel`:
it verifies the narrower legality of moving one resource read in time.

```text
view load definition ── capture buffer, origin, fill and bounds facts
             │ no intervening writes or stage boundary
             ▼
first reduction loop ── load element once ── contribution ── partial
                              │
                later users?  ├── yes: save snapshot element
                              └── no: omit snapshot allocation
```

The loaded scalar is reused by repeated pointwise operands such as `x*x`.
When later consumers exist, they still read the original snapshot, including
after an aliasing store. Captured XIR SSA definitions are immutable; the
host-side pending plan is consumed at the reduction rather than reused across
repeated lowering of a loop body. Bounds guards/fill share the ordinary view
access emitter. No reciprocal rewrite, invented reduction identity or tree
change is introduced.

The work prior charges the external read once, removes private reads in the
fused first consumer, and removes snapshot writes only when no later consumer
needs storage. It uses the **same admission helper** as lowering. This is
realization-derived work accounting, not measured cycle calibration or a
general phase-fusion solver. Multiple-use math recipes, nonunit wrapper maps,
arbitrary gather consumers and cross-effect reloads remain unoptimized here.

The SIMD diagnostic switch `LUISA_SIMD_ENABLE_LOAD_REDUCTION_FUSION=1`
enables both planning and lowering of this rule for fixed-mapping A/B tests;
`LUISA_SIMD_DISABLE_LOAD_REDUCTION_FUSION=1` takes precedence and disables it.
Measured RMSNorm regressions show why fewer counted private reads alone do
not justify enabling this rule by default. See the
[performance evidence](../../performance/tile/results.md#simd-local-distribution-and-private-layout-are-separate-decisions).
`fused_reduction_loads` and `elided_load_snapshots` are static construction
counts in realization metadata, not dynamic memory-transaction counts.

### Packet-private storage budgets

The SIMD adapter separately budgets **physical packet storage**:
`bytes = Σ align_and_place(W × sizeof(local_array))`. It keeps small private
arrays on the stack; above 64 KiB it moves their distinct intervals into a
64-byte-aligned, Runtime-owned CPU-thread workspace, capped at 16 MiB.
The allocation grows on first use and is reused after each packet completes;
separate CPU threads never share it. It does not escape through kernel
parameters or change user Buffer layouts. Standalone SIMD compilation retains
the old stack ABI unless this policy is explicitly enabled. Cooperative
packets and nested handlers cannot use this reuse policy.

This is a physical-allocation constraint, **not a measured complete stack
bound**, peak-liveness solver or guarantee against arbitrary register-spill
growth. Standalone lowering defaults to a 256 KiB logical-worker budget; the
Runtime adapter derives that budget from `16 MiB / W` and codegen checks the
final aligned placement against the physical capacity again.
Both counts and the chosen representation controls appear in realization
metadata. Root order and block width participate in default exact search;
local-axis distribution can be included explicitly. Small-loop regressions and remaining resource rejections
must be assessed separately from successful large-kernel compilation.

### Packet-local distribution preserves the split coordinate

For an admitted common local axis of extent `N >= W`, the bridge can assign
one root program to a complete packet:

```text
source: root program c, element e             0 <= e < N
                   │
        u = flatten_pi(c)
        e = W*q + lane                      0 <= lane < W
                   │
                   ▼
physical worker = W*u + lane
private element = q                         tail: W*q + lane < N
```

This is a realization of the existing program, not a new DSL scope or an
extra independence assertion about `parallel`. The lowerer retains `(q, lane)`
as a split coordinate: projecting an owner-preserving Tile reads slot `q`
directly. Reconstructing `q` with varying-i64 division after flattening loses
valuable structure before Schedule/LLVM even sees the program.

The current sufficient admission contract requires one shared **dimension
identity** and extent across all nonunit Tile/map/reduce axes. Unit axes may
be inserted or projected. Extracts preserve that axis coordinate; arbitrary
permutations, cross-lane indexing, nested varying axes, strict folds, complex
carry, explicit execution binding and manual Memory keep the whole-program
fallback. Forced unsupported distribution fails closed. These limits describe
missing realizations, not dependencies supposedly absent from the language.

Loads normally snapshot at their definitions; the opt-in fusion above may
move initialization to the first reduction. Each lane owns `ceil(N/W)` private
slots, with only valid tail slots accessed. A closed unordered reduction starts
each lane's partials from real contributions, merges them with a fixed
`WARP_READ_LANE` butterfly, broadcasts lane zero's tree root, and combines the
source initial value exactly once. Every lane reconverges before a shuffle;
unit output stores execute only on the packet leader. No new zero/one identity
or global fast-math permission is introduced.

The output metadata carries `required_packet_width`: zero for whole-program
lanes, exactly `W` for packet-local programs. The SIMD adapter checks this
contract before compilation. Dispatch has `P*W` physical workers, so a logical
program is never launched as a partial packet; the final Runtime block may
still contain fewer complete packets.

### Private array layout is independent of execution distribution

The SIMD backend can independently interleave a nonescaping scalar array:

```text
                         whole-program or packet-local execution
                                           │
logical private access (lane, q)            │
                  ├── lane-major:    lane * array_length + q
                  └── interleaved:   q * W + lane
                                           │
                            stack or CPU-thread workspace
```

This backend transformation is not keyed on a Tile operator name. It admits
only the closed address tree `alloca -> typed scalar-element GEP -> load/store`
for 4/8-byte scalar elements. Aggregate access, reference escape, address PHIs
and shared memory retain the original layout. The bijection preserves every
lane's distinct storage, snapshots and capacity; it does not merge lifetimes
or reorder effects. Interleaving alone does **not** guarantee that the emitter
recognizes a contiguous masked access or that LLVM removes address overhead.

The Tile adapter enables this representation and reports
`interleaved_private_arrays`; standalone SIMD compilation keeps it opt-in.
`LUISA_SIMD_DISABLE_INTERLEAVED_PRIVATE_ARRAYS=1` is a diagnostic A/B control.
Execution mapping and private layout must be measured separately: a favorable
layout does not make all packet-local executions profitable.

### Common-slot private accesses preserve the scalar allocation base

For an admitted interleaved allocation, a common slot has the address family
`base + sizeof(T) * (q * W + lane)`. The backend now retains the allocation
identity alongside each eligible access. The allocation base is immutable
and dominates its uses; the offset comes from the **saved GEP handle**, not
from re-evaluating an index after a loop or a scheduler transition.

```text
closed private allocation ─── immutable scalar base
saved GEP + active cohort ─── common q * W * sizeof(T)
                                          │
                               one complete private slot
                                  [lane 0 ... lane W-1]
                                          │
                     read vector / preserve inactive store bits
```

A warp-uniform index qualifies across Schedule blocks. A cohort-uniform
index qualifies only when its GEP and access are in the same Schedule block;
cohort equality is not a claim that values stay equal across reconvergence
or suspension. Varying indices and non-admitted address trees retain the
gather/scatter path. The existing seed of the current cohort is reused; an
empty cohort selects allocated slot zero rather than an invalid inactive
handle.

Unlike external or shared memory, every complete slot of this closed
packet-private allocation has storage for all W lanes. A vector load may
therefore read that slot and select inactive lanes to zero. A partial store
loads the previous slot, selects new values for active lanes, and writes the
vector back, preserving every inactive bit. This is safe only because the
allocation has no escaping aliases or concurrent observers. It does not
authorize external-buffer overreads, wider shared-memory writes, snapshot
reordering or lifetime coalescing.

`contiguous_private_reads` and `contiguous_private_writes` count statically
emitted eligible accesses, not executed memory operations or calibrated
cost. Region versioning may emit more than one realization of an access.
`LUISA_SIMD_DISABLE_CONTIGUOUS_PRIVATE_ACCESS=1` holds execution mapping and
private layout fixed while restoring the gather/scatter control. Actual
target code still decides profitability: a masked-vector intrinsic alone
does not guarantee native vector instructions on a target without predicated
loads and stores.

### Proven packet accesses, not estimated slopes

The SIMD Schedule projection separately recognizes a bounded nonnegative
integer expression of the form `x[lane] = W*q + lane`. Its initial seeds are
packet-aligned dispatch/thread x coordinates or the lane index. Only
power-of-two widths and packets contained in one x row receive this proof.
Value-preserving integer casts, aligned constant offsets and supported
quotient/remainder compositions retain it; unknown expressions fail closed.

For positive `d` divisible by W, an aligned packet cannot straddle a d-sized
boundary. Therefore `x/d` is cohort-equal and `x%d` is consecutive. Division
by one preserves x; remainder by one is equal to zero. A truncated cast,
unaligned offset, ragged divisor, negative/unknown divisor or cross-row packet
does not satisfy this rule. Same-width signed/unsigned i64 casts preserve
address bits; they do not authorize narrow wrap followed by extension.

```text
uint32 dispatch x → proved value-preserving i64 cast
                           ├── / aligned row width → same row → broadcast A
                           └── % aligned row width → adjacent columns → load B
                                      ↓
                         existing masked SIMD memory emitter
```

These annotations reach the existing nonvolatile buffer broadcast/contiguous
load/store implementation. Active masks, first-active-lane addressing and
tail guards remain in force. They neither move memory effects nor relax MMA
arithmetic. This closes a specific disconnect where the planner estimated
coherent accesses but the emitter still generated gathers; it does not add
cache blocking or a packed matrix atom. See the
[frozen compiler comparison](../../performance/tile/results.md#simd-packet-index-proof-closes-a-codegen-disconnect).

## 4. The implemented solver

### Candidate space and hard constraints

The first solver searches the Cartesian product of:

- All permutations of root parallel axes, unless an exact order is supplied.
- Block worker counts `{32, 64, 128, 256, 512, 1024}`, unless fixed explicitly.
- With `local_lanes=0`, whole-program lanes and an admitted full-packet local
  axis; `local_lanes=W` fixes the latter. The default remains `local_lanes=1`
  until tail-control and CPU worker-activation costs are modeled adequately.

The target packet width is an existing Device property, not a compiler guess.
Block counts must satisfy XIR's block-size contract and be divisible by that
width. Root domains must be static, nonempty, uint32-addressable and independent;
there must be one root parallel with no escaping state. Supported descendant
regions retain their existing local order. Unsupported operation, explicit
binding or manual resource requirements are rejected.

The solver enumerates the entire declared finite space and returns its exact
minimum **under the specified cost function**. It is not a globally optimal
hardware scheduler. The default budget is 1024 candidates; exceeding it is an
error asking for tighter constraints, not silent partial search. Ties are
deterministic. Input IR verification and the lowerer's own legality checks
remain authoritative; a low score never makes unsupported code legal.

### Cost units and formula

`ExecutionCostModel` is an **uncalibrated relative-work prior**, not nanoseconds,
hardware instruction counts or measured cache behavior. Default weights:
arithmetic 1, broadcast load 1, contiguous memory 2, gathered lane 2, block
dispatch 128. All coefficients must be finite and nonnegative.

For each candidate, the estimator counts Tile work, local-loop
repetition, ordered MMA multiply/add work, definition-time snapshot stores
and runtime indexed reads. The planner and lowerer share the structural
classification of expanded versus runtime Tile-extract coordinates. The
indexed-read prior includes flat-index/guard arithmetic and gathered local
memory; it no longer charges an entire Tile selection for every iteration.
The representation classifier also accounts for bounded-array reads/stores,
deferred recipes at the consumer, and staged copies of large carries. Partial
accumulator counts and the representation threshold are configurable fixed
constraints, not newly searched or empirically calibrated dimensions.
These are still relative-work estimates, not exact machine instruction counts.
It estimates a buffer's flat address slope relative to the innermost root
axis, using operand identity and supported constant/linear expressions.
Slope zero on a load has a broadcast prior; absolute slope one has a
contiguous prior; other or unknown addressing pays the gather prior times W.
An innermost extent not divisible by W conservatively doubles memory work.
These classifications are **not passed to codegen as proven facts**.

Let `a,m` be estimated arithmetic/memory work per packet, `H` available CPU
workers, `P` physical workers (`root programs * local_lanes`), `Q=ceil(P/W)`, `L=ceil(P/B)`,
`h=min(H,L)`, `waves=ceil(L/h)`, and `d` the dispatch weight:

```text
arithmetic = a × Q / h
memory     = m × Q / h
dispatch   = d × waves
imbalance  = max(0, waves × ceil(min(P,B)/W) − Q/h) × (a+m)
score      = arithmetic + memory + dispatch + imbalance
```

All four terms are retained in the plan and reported in shader realization
metadata, along with the selected order and candidate count. This homogeneous
wave model intentionally does not pretend to model the M1's heterogeneous
cores, cache sharing, variable mask density, spills or actual thread timing.
The experimental distribution estimator counts local iterations, external
access slopes and a fixed shuffle prior. Its private-array estimate remains
conservative and uncalibrated; it does not yet price the interleaved emitter's
actual accesses. Joint search is therefore opt-in, not a promised speedup.

### Reproducible fixed-plan controls

```cpp
bridge::xir::PlannerOptions options;
options.block_size = 64;
options.root_axis_order = {0, 1}; // outer-to-inner; exact, not a hint
auto shader = tile::compile(device, kernel, {.xir = &options});
```

`CompileOptions::threads_per_group` and the XIR block constraint must agree
when both are supplied. Configuration is borrowed only during synchronous
compilation. Metal rejects XIR options rather than silently ignoring them.

Searching a physical plan is distinct from tuning the C++ specialization:
changing BM/BN/BK or the semantic pipeline shape simply recaptures the lambda.
An outer JIT tuner can search these variants, using the same correctness and
measurement gates. There is no capture-once restriction.

## 5. Lowering invariants and supported behavior

| Tile semantics | XIR realization |
|---|---|
| Tile value | Small SSA, bounded indexed array or single-use pure recipe; whole-program or admitted packet-local distribution |
| Named dimensions | Identity-based projection/broadcast; names are diagnostics |
| Load snapshot | Load at the source operation before subsequent effects |
| Bounds/fill | Per-axis guards; actual load executes only in the valid branch |
| Store | Explicit guarded buffer effect, including BufferView offsets |
| Loop-carried assignment | Small header PHIs or large staged parallel copy; zero-trip initial state preserved |
| Pipeline/stage | Ordered CPU loop and source-order phase cuts; no claimed physical overlap |
| Reduction | Closed unordered single-carry partials, optionally packet shuffles; strict/non-closed fallback retains order |
| MMA | Ordered multiply/add traversal with initial accumulator and dimension contraction |
| `ite(c,t,f)` | Correctly reordered to XIR's `SELECT(f,t,c)` |

The checked expansion budget defaults to 262144 values. Supported scalar
types are bool, i32/u32, i64/u64, f32/f64; fp16/bf16 are not yet supported by
this lowerer. Cooperative group/subgroup bindings, explicit manual Memory,
arbitrary resource/address mappings and multi-launch programs remain outside
the implemented subset. SIMD vectorization of workers is not tensor-core or
matrix-extension lowering.

Candidate TileIR still retains pure multi-consumer SSA definitions. The direct
XIR bridge does not yet search recomputation versus a distributed physical
materialization; bounded traversal, single-use recipes and the existing
XIR/SIMD shared-SSA cleanup are a structural policy. A future XIR resource candidate must use
the same use/effect/ownership facts as TIRx, but may choose a different result
for CPU SIMD. It must not infer a user `Memory` or mechanically copy Metal's
worker-stripe policy.

## 6. Why not split every Tile into more workers?

Changing the enumeration of independent root programs is safe. Introducing
new worker boundaries *inside* one program needs additional justification.

```text
One original worker:
  x = input.tile(...).load();   // snapshot of all elements
  output.tile(...).store(x);   // input/output may overlap

Naive split:
  worker 0 loads and stores its element
  worker 1 may load after worker 0 overwrites its source
```

The second program can violate the first program's semantics even if output
coordinates are distinct. Const input views are not noalias promises.
Reductions, dynamic extraction and shared loop-carried state introduce further
dependencies. Thus a general distribution candidate must carry a dependence
and effect analysis, a collective realization, or a checked invocation contract
with a safe fallback. Shape alone is insufficient.
The packet-local candidate above preserves complete definition-time loads
before stores; it does not perform this naive per-element load/store fusion.

## 7. Extension plan: richer plans, not more DSL entities

```{figure} ../../../_static/tile/xir-mapping-roadmap.svg
:alt: The implemented root-order and packing search precedes future Tile partitioning, collective atoms, resource planning and physical pipelining.
:width: 100%

Dashed boxes are planned extensions. Every new search family needs a supported emitter and its own correctness obligations.
```

New candidate families enter only when their emitters and proof obligations
exist. A small space uses exhaustive search. A larger factorized space may use
dynamic programming or branch-and-bound; ILP/CP-SAT is useful for discrete
resource and dependence constraints; annealing/beam search may propose
profitable candidates. Approximate solvers must report budget, explored space
and absence of a global guarantee. None is implemented by merely adding the
algorithm's name to a cost model.

Calibration should measure independent mechanisms and retain uncertainty:
packet memory coherence, masked work, dispatch batching, arithmetic mix,
working sets, duplicated loads/recomputation, live shared values, compile size
and spills. Evaluate top-choice regret, top-K
coverage and JIT cost on held-out shapes **and held-out operator families**.
Do not fit a GEMM-only model and label it an LLM model. Cache keys must include
IR specialization, plan schema, compiler/device identity, numerical policy and
cost-model revision; structural transforms invalidate affected plans.

A future Machine TileIR should expose typed realized atoms, execution maps,
resource instances, layouts, lifetimes and synchronization so passes can
inspect and rewrite them. It should not duplicate frontend math or become a
serialized backend instruction list. The current `ExecutionPlan` and XIR
Module are concrete, smaller stepping stones, not a claim that this full
intermediate representation already exists.

### Bounded local-vector candidates

**Local-vector distributions remain proposed; snapshots, bounded traversal
and closed unordered partials are implemented.** The
[Torch CPU code inspection](../../performance/tile/results.md#torch-cpu-code-inspection-exposes-missing-local-vector-candidates)
identified static expansion and dynamic selection chains in the previous
bridge's machine code. The indexable snapshot repair above addresses the
selection representation, and bounded traversal removes whole-row static
expansion. Neither solves the whole distribution problem: the next candidate
family needs independent output/contribution partition factors.

For logical packet width W and p lanes per independent output, p dividing W,
a local candidate maps lane l to `(o0 + floor(l/p), r0 + l%p)` and advances
`r0 = t*p` over time. It covers vectorizing across outputs (p=1), across
contributions (p=W), and mixed packing. Physical vector width, output grain,
unroll and scratch choices remain target policy decisions. A phase may use a
different partition from its successor and must account for the transition.

Closed unordered reductions may use partial accumulators and horizontal
combination. Strict folds retain their required contribution order but can
still vectorize independent outputs. Replacing a load snapshot with a view
requires actual alias/effect conditions; `parallel` supplies independence
between its instances, not permission to change the effect order inside one.
Compiler storage is not a new user Memory obligation.

Cost calibration follows implementation: distinguish gather/contiguous work,
vector math, horizontal combine, masks, peak live state and spills, phase
transitions and CPU grain. Use separate IR/code-size and JIT budgets to prevent
static expansion from overwhelming compilation. Reuse the SIMD backend's
existing fixed-vector math provider. Neither arbitrary lane widening nor a
new solver algorithm can substitute for a realizable local-vector family.

## 8. Validation entry points

- `test_tile_xir`: typed ABI, output verification, repeat lowering, bounds on
  expansion, unsupported bindings, permutation legality, exact minimum and
  fixed-plan/budget failure cases; linear snapshot construction, zero SELECTs
  for proved static projections, exact local-storage boundaries and constant
  XIR size from width 65 through 16384 for the bounded sumsquares fixture.
- `test_tile_xir_runtime`: ragged/transposed GEMM, nonzero initial values,
  changed non-dyadic inputs, reductions/softmax, offset views, guards, shader
  moves, zero-trip loops and read/write snapshot recurrences, including
  dynamically indexed multi-element carry swaps and aliased const/writable
  buffer arguments; large in-place transpose across Runtime workers and
  partial-reduction seed/tail/signed-zero/non-closed-fallback checks.
- `test_tile_xir_llm`: normalization, activations, RoPE, masked softmax and
  online prefill/decode/GQA; same capture through XIR and native-target TIRx,
  each checked independently against an FP64 oracle. Separate large Runtime
  workspace fixtures bypass the TIRx cross-check but retain full FP64/guards.
- `test_simd_phi_parallel_copy`: pure PHI cycles, uniform/varying loops,
  packet widths 1/2/4/8/16 and every active-lane count, independent of TileIR.
- `benchmark_tile_xir`: isolated warm host-wall timing, full output export,
  realized plan and LLVM identity, with planned/canonical/reversed controls.

These are validation mechanisms, not by themselves performance results. See
the [status and evidence report](../../performance/tile/index.md) for actual runs,
limitations and links to raw evidence.
