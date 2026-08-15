# SIMD native execution contract

Status: formal boundary and executable-audit plan for instruction and
device-library lowering.

The scheduler model specifies *which lanes* execute an instruction. This
document specifies *how one scheduled instruction* may be represented without
silently turning a varying operation into `W` scalar operations. The two
models are complementary: a correct cohort mask does not make a per-lane math
loop SIMD-native, and a vector math routine does not make an unmasked side
effect correct.

## 1. State and observation

For packet width `W`, let `L = {0, ..., W - 1}` and let `A` be the current
active-lane mask. A source instruction has the abstract form

```text
(A, x_0, ..., x_n, M) -> (r, M')
```

where `M` is externally observable memory and resource state. A varying value
is a lane map `x : L -> T`; a warp-uniform value is one scalar `x : T`.
Results outside `A` are not observed unless they are later filled by another
cohort through a masked state-slot merge.

The refinement relation requires, for every `l in A`:

```text
r_llvm[l] = r_scalar(l)
M'_llvm restricted to effects of l = M'_scalar(l)
```

No equality is required for `r_llvm[l]` when `l` is not in `A`, but evaluating
such a lane must not trap, access an invalid address, call an effectful
function, or introduce LLVM poison that can affect an active observation.

## 2. Representation and uniformity

The representation function is

```text
R(warp_uniform<T>) = T
R(varying<T>)      = <W x T>
R(mask)            = <W x i1>
```

Kernel value/resource parameters, constants, block ID, dispatch size, block
size, kernel ID, and warp size are warp-uniform seeds. They remain scalar
through device-library and math calls. A scalar is splatted only at a varying
consumer. In particular, a uniform `sin(x)` is one scalar operation, not a
vector call with `x` broadcast to all lanes.

Callable arguments are specialized from their call sites when possible. An
unspecialized callable argument remains conservatively varying; it may not be
silently converted into a scalar merely because one current caller happens to
pass equal lane values.

## 3. Inactive-lane safety classes

Each lowered operation belongs to exactly one class:

| Class | Examples | Required treatment outside `A` |
| --- | --- | --- |
| total pure | add, xor, finite compare | result may be arbitrary |
| partial/trapping | integer div/rem, shifts, float-to-int, table index | sanitize operands before evaluation |
| memory read | buffer gather, math-table gather | masked access or an in-bounds sanitized address |
| side effect | store, atomic, print, assert, trace | predicate the effect by `A` |
| collective | vote, reduce, shuffle | use its explicit participant mask |

For a partial operation `f`, codegen must choose neutral operands `n_i` such
that `f(n_0, ..., n_k)` is defined, then evaluate

```text
x'_i = select(A, x_i, n_i)
r    = f(x'_0, ..., x'_k)
```

A masked merge after `f` is insufficient: host integer division can trap
before that merge, and an out-of-range table gather can fault. The permanent
partial-tail remainder regression in `test_simd_runtime_widths` fixes this
distinction.

## 4. Native-lowering predicate

Let `V(op, W, target)` be the machine implementation selected for one varying
source operation. `native(op, W, target)` holds when all of the following are
true:

1. the backend source contains target-independent fixed-vector IR for the
   operation or one call to a verified vector ABI;
2. it contains no loop whose induction variable enumerates lanes;
3. it contains no `extractelement`/scalar-call/`insertelement` sequence for a
   device-library function;
4. final code has no unresolved scalar device-library symbol attributable to
   the varying operation;
5. the selected ABI is available on the process and legal for the detected
   CPU features;
6. active-lane numerical and exceptional-value semantics satisfy the declared
   accuracy tier.

LLVM fixed-vector syntax alone does not prove this predicate. For example,
LLVM may legally lower `llvm.sin.v8f32` to eight scalar `sinf` calls. The
object/assembly audit is therefore part of acceptance, not a benchmark-only
check.

Target legalization may split `<16 x float>` into two or four physical vector
instructions. That still satisfies `native`: the split is by physical vector
width, not a hidden source-level lane loop or scalar device-library call.

For reproducible host scheduling diagnostics, a device whose
`SIMDDeviceConfigExt::worker_count()` is zero may read a positive decimal
`LUISA_SIMD_WORKER_COUNT`. An explicit nonzero API value always wins, including
when the environment is malformed; otherwise zero, malformed, or trailing
characters fail closed. This changes only the persistent host worker-pool size,
not warp width, packet ABI, block ordering, or kernel semantics.

### Pre-schedule aggregate promotion

The AST compiler front door runs target-independent XIR SROA before each of
its two `mem2reg` stages. The SIMD policy decomposes local structs and arrays
one level at a time; it does not request vector or matrix decomposition. An
alloca is eligible only when it is `LOCAL`, every transitive use is a load,
store, or GEP, and the first index of every directly rooted GEP is a constant
in range. Dynamic indices below that selected member are preserved. Unknown or
escaping uses, dynamic first indices, and instruction metadata that cannot be
mapped uniquely reject the alloca before any replacement is inserted.

The transform preserves source semantics and inactive-lane rules: it changes
only private storage identity, clones semantic storage metadata, and leaves
masking/sanitization to the same downstream Schedule lowering. It does not
split resources, atomics, ray-query objects, external ABI aggregates, or
memory reached through pointers. The resulting leaf allocas may be promoted
to scalar or fixed-vector SSA; no contract requires LLVM to keep every leaf in
a physical register.

`LUISA_SIMD_DISABLE_AGGREGATE_PROMOTION=1` disables both pipeline invocations
for differential diagnostics. `LUISA_SIMD_REPORT_OPTIMIZATIONS=1` reports
`aggregate_allocas` and `aggregate_leaf_allocas`. Permanent execution coverage
compares enabled and disabled output at W1/W2/W4/W8/W16 across varying partial
field updates, loop-carried values, and inactive tails.

### 4.1 Direct coherent CFG refinement

For `W = 1`, a warp has no distinct lanes that can diverge or rendezvous.
Schedule-to-LLVM therefore emits the ordinary scalar CFG directly: an initial
dispatch-bounds test guards the entry block, control-edge assignments preserve
PHI/state semantics, and return exits the JIT entry immediately. The width-one
path does not allocate or execute lane-PC, convergence-token, frame, runnable,
or loop-epoch scheduler state. Cross-block Schedule values still use the
verified state-slot representation before LLVM optimization, allowing
`mem2reg` to recover scalar SSA on the direct CFG.

This is an implementation refinement of the cohort model, not a separate
semantic mode. Inactive dispatch-edge invocations perform no instruction or
resource side effect; resource arguments, local storage, atomics, math, and
the packet return ABI remain shared with wider specializations. W2/W4/W8/W16
refine the independent-lane model with the bounded scalar-target/vector-mask
worklist described by the scheduler formal model.

The same refinement applies at W2/W4/W8/W16 when static uniformity proves that
the whole reachable Schedule CFG cannot split a cohort: every conditional or
indexed selector is `warp_uniform` or `cohort_uniform`. The LLVM function then
contains direct scalar branches around fixed-vector values. Its active mask is
the immutable dispatch-tail mask; cohort-uniform state remains scalar across
blocks, and varying memory/effects remain predicated by that mask. A
cohort-uniform selector may choose a different edge for different packet
invocations, but never chooses different edges for active lanes in one packet.

A second, deliberately local refinement permits a convergence point only when
its complete varying diamond satisfies the predicated-memory rules in Section
4.4.1. Both arm blocks and that convergence are then eliminated from emitted
LLVM control flow. Whole-function direct lowering is allowed only if every
remaining convergence point has been eliminated this way and every other
terminator satisfies the coherent proof.

This proof is fail-closed. Any varying selector or convergence point not
covered by Section 4.4.1, or any unsupported terminator, retains the general
worklist scheduler. Separately, the general scheduler may discover at runtime
that every active lane of one varying branch chose the same successor and
directly thread that edge; this does not make subsequent control statically
direct or discard scheduler state.

On that dynamically coherent edge, codegen passes the incoming active mask to
the successor instead of the algebraically derived branch/case mask. The
runtime all/none/mixed test proves the two masks equal before this substitution:
conditional true/false masks partition the nonempty incoming mask, and indexed
case/default masks form the corresponding disjoint partition. Edge assignments
therefore retain exactly the same active lanes, including a partial dispatch
tail. A genuinely divergent partition never takes this substitution. The
diagnostic oracle `LUISA_SIMD_DISABLE_COHERENT_MASK_REUSE=1` restores the
derived masks, while `coherent_mask_reuses` in the optimization report counts
eligible static successor edges. Permanent tests execute coherent true/false
and every indexed case/default at W2/W4/W8/W16, compare the oracle, cover
partial tails and one-lane cohorts, and retain a genuinely divergent switch.

#### 4.1.1 Canonical early-exit header routing

The general scheduler may avoid partitioning one varying counted-loop header
when Schedule lowering attaches `cohort_uniform_condition`. This annotation is
valid only for the direct comparison recognized by the canonical loop-bounds
analyzer after a local analysis view removes non-header early-exit edges. The
source loop must retain one preheader and latch, an integer induction PHI,
constant nonzero stride, and uniform start and bound. The backing PHI and
predicate remain lane-wise `varying`; the annotation authorizes no scalar spill
and no use outside that terminator.

The executing continuation is keyed by the relevant natural-loop epoch, so
all of its active lanes have the same induction value. Codegen first forms
`select(active_mask, condition, false)` and only then performs `or.reduce`.
The inactive lanes are therefore benign before the reduction, including when
their stored predicate bits are stale or poison. Because the executing mask is
nonempty and every active predicate agrees, the scalar reduction selects the
same unique edge as the ordinary two-mask partition. The entire incoming mask
is passed to that edge; no convergence frame is allocated merely for this
coherent header decision.

The split's convergence and edge-join metadata must remain present. A sibling
early exit can park lanes from an earlier epoch, and the eventual normal exit
must still complete the same post-loop rendezvous and collective instance.
Failure of any structural, recurrence, or uniform-bound proof falls back to
the ordinary varying path. A 25-Schedule-block minimum is a performance policy,
not part of correctness. `LUISA_SIMD_DISABLE_COHORT_UNIFORM_INDUCTION=1`
restores the generic lowering in the same binary, and
`cohort_uniform_loop_branches` reports accepted general-scheduler sites. W1
continues to use its direct scalar CFG.

Permanent differential coverage uses a 25-block loop, lane-dependent exits in
different epochs, two exit destinations, a post-loop `warp_active_sum`, W1/W2/
W4/W8/W16, inactive output sentinels, LLVM verification, and exact candidate/
oracle results. Separate structural cases reject a 24-block loop and a
lane-varying bound.

On a genuinely divergent binary split, the worklist is LIFO: after the true
and false records are appended, the false record is necessarily the next
record removed. At W4/W8/W16 and at least 32 Schedule state slots, codegen may
eliminate that false-record push/pop pair. It appends the true record, keeps
the current scalar token, stores the false mask as `current.mask`, and passes
the constant false target through the same `scheduler.dispatch.route` and
dispatch switch used by normal pops. Both edge-assignment sets have already
executed under their disjoint masks, `runnable.mask` contains exactly the
suspended true lanes, and the false destination retains its ordinary dynamic
convergence arrival. No Schedule block is entered directly and no dispatcher
switch is cloned.

This refinement does not change the supported control-flow set. W1/W2 and
functions with fewer than 32 state slots retain the explicit worklist
sequence. `LUISA_SIMD_DISABLE_DIRECT_DIVERGENT_CHILD=1` is the production
same-binary oracle; `LUISA_SIMD_FORCE_DIRECT_DIVERGENT_CHILD=1` is a diagnostic
force for low-state W4/W8/W16 fixtures; `direct_divergent_children` counts
emitted sites. Permanent coverage executes every active-lane count from zero
through the width with inactive sentinels and a branch-local warp collective,
compares candidate/oracle results at W4/W8/W16, and requires exact W2 IR
identity.

`LUISA_SIMD_DISABLE_COHERENT_DIRECT_CFG=1` forces the scheduled implementation
for differential diagnostics. Permanent tests cover all widths, partial tails,
cohort-uniform branches with packet-dependent outcomes, and the forced fallback.

#### Scheduled PHI physical-slot coalescing

The general scheduler retains every logical Schedule value and its parallel
edge assignment, but LLVM storage may coalesce noninterfering PHI state
versions. Codegen first solves exact backwards liveness over every verified
Schedule edge. The first stage considers a source/destination pair only when
both are state slots with the same nonempty XIR-derived name, uniformity class,
and LLVM storage type, and no values in their proposed groups interfere. Names
limit move candidates; they are not a correctness proof. Each destination also
interferes with all other parallel-copy sources on that edge, so deterministic
assignment emission cannot overwrite an unread source.

For high-pressure W16 schedules, a second stage may color compatible non-move
roots. Roots are ranked by a precomputed interference degree and greedily
joined only when the complete groups do not interfere and their value class,
Luisa type, local-lvalue status, and allocated LLVM type agree. Production
requires at least 32 logical state slots and at least two additional eliminated
physical slots. Failing the latter test restores the exact first-stage parent
map, so a one-slot opportunity cannot perturb production code. W1/W2/W4/W8 do
not run the second stage.

Liveness is defined per logical lane. Suspended divergent cohorts may reside
at different PCs, but Lane ownership gives them disjoint masks, so different
components of one fixed-vector slot may represent different nonoverlapping
versions. A coalesced source/destination assignment is an identity for every
participating lane and is omitted. No lane mask, logical PHI value, token,
epoch, memory effect, or supported CFG changes. W1 and statically direct CFG
do not run the transform.

`LUISA_SIMD_DISABLE_STATE_PHI_COALESCING=1` restores distinct physical allocas
and explicit masked moves for both stages.
`LUISA_SIMD_DISABLE_GENERAL_STATE_COLORING=1` is the production same-binary
oracle for the second stage; `LUISA_SIMD_FORCE_GENERAL_STATE_COLORING=1`
bypasses only its width, state-count, and two-slot profitability gates. The
optimization report exposes total eliminated storage as
`coalesced_state_slots`, the general-coloring subset as
`general_colored_state_slots`, and retains `state_slots` as the logical count.
Permanent W1/W2/W4/W8/W16 coverage compares enabled/disabled execution for
every active-tail length, includes a state-to-state passthrough, a two-value
parallel-copy swap, two separate high-pressure live ranges, and a one-slot
rollback case. It requires inactive sentinels, verifies direct-W1 and
production W2/W4/W8 assembly identity, and requires the intended W16
code-shape difference. Final object checks continue to reject varying
scalar-libm calls.

For a scheduled destination that can be a convergence target, scalar token
zero proves that the current cohort has no dynamic frame. The target-arrival
cascade must then preserve the incoming mask and all scheduler state, so the
LLVM lowering bypasses frame target, expected/arrived, active-frame, parent,
and runnable accesses. When a completed frame restores parent token zero, the
lowering also omits the otherwise guaranteed identity iteration at the end of
the chain. Nonzero parents retain the full dynamic target comparison and may
release further same-target frames. The diagnostic oracle
`LUISA_SIMD_DISABLE_CONVERGENCE_TOKEN_GUARD=1` disables both shortcuts, and the
optimization report exposes `convergence_token_guards`.

Permanent candidate/oracle execution coverage uses W2/W4/W8/W16, every active
lane count from one through the width, inactive sentinels, ordinary root-token
arrivals, nested parent/child frames sharing one target, and early return that
completes the cascade. The unoptimized module is retained as a semantic oracle;
the optimized module must also contain the named token guard and pass LLVM
verification.

After a scheduled return removes the terminating cohort from `live` and
`runnable`, an all-zero `frame.active` bitset proves that there is no frame
whose expected set can shrink or whose arrived lanes can be released. The
per-frame return cleanup and every zero-mask ready resume are then identities.
LLVM emission tests the scalar bitset once and bypasses the complete bounded
W-frame cleanup when it is zero; a nonzero bitset enters the unchanged cleanup,
including parent-token restoration and release of lanes parked by an early
return. `LUISA_SIMD_DISABLE_RETURN_FRAME_GUARD=1` restores the unconditional
cleanup, and the optimization report exposes `return_frame_guards`.

Permanent W2/W4/W8/W16 coverage compiles and executes both forms for every
active-tail length. Its nested same-target fixture reaches one return while
frames are live and a later return after they have been released, checks
inactive sentinels and exact results, verifies both LLVM modules, and requires
the named `return.frames.present` guard only in the optimized form. Direct W1
and statically coherent CFGs allocate no frame state and report zero guards.

The physical representation of convergence-frame static IDs and parent tokens
is independently selectable without changing those token/cascade semantics.
W1/W2/W4/W8 use `<W x i32>` storage. W16 uses `[16 x i32]` plus scalar
GEP/load/store at the checked dynamic frame index, reducing dynamic whole-ZMM
updates and register pressure. Active-frame validity is established before the
access, and the no-token path supplies index zero only to control flow that is
otherwise masked or bypassed; the array form introduces no speculative
out-of-bounds access. `frame.active`, expected masks, arrived masks, ready
records, and convergence-target lookup are unchanged.

`LUISA_SIMD_DISABLE_SCALAR_FRAME_METADATA=1` is the same-binary W16 oracle.
The runtime optimization report exposes `scalar_frame_metadata`; direct CFG
and W1/W2/W4/W8 report false. Permanent coverage uses a nested same-target
convergence cascade with early return, all active-lane counts from zero through
W, inactive sentinels, exact candidate/oracle output equality, W16 array versus
vector IR/assembly checks, and byte-identical W1/W2/W4/W8 IR/assembly. The
width restriction is performance policy: W4 and W8 scalar arrays failed paired
throughput gates, while W16 passed both compiler-control and real-example
gates.

Permanent code-shape and execution regressions cover a divergent diamond, a
natural loop, a switch inside a loop with early exits, and an inactive W1
dispatch. The unoptimized W1 module is rejected if it contains
`scheduler.loop`, convergence-token, or frame-state storage.

### 4.2 Host block concurrency

A shader dispatch is synchronous at the stream boundary but parallel across
thread blocks. The device-owned persistent worker pool dynamically partitions
the flattened block interval `[0, grid_size.x * grid_size.y * grid_size.z)`.
All packet calls for one block execute serially in increasing warp order on a
single worker; different blocks may execute concurrently and in any order.
The pool joins before the next command, dispatch size, download, or callback in
the same command list becomes observable.

For W2/W4/W8/W16 blocks with more than one packet, the production compiler may
replace the repeated host-to-JIT calls by one exclusive block-local packet-batch
entry. Its first three arguments retain the ordinary physical packet ABI and
its fourth argument is the number of full logical packets. Starting from the
block-owned `launch_config.thread_index`, packet `p` executes the ordinary body
with first thread `base + p * W`. Packets remain strictly ordered. The generic
form passes active-lane count `W` and retains the body's exact dispatch checks.
For a runtime static block `{X, 1, 1}` with `X % W == 0`, the wrapper instead
computes the dispatch/block remainder in 64-bit arithmetic, executes only the
complete all-on packets, and issues at most one narrowed prefix tail before
any packet operation. An empty suffix is not called. This makes the packet
prefix itself the exact dispatch mask and does not extend the kernel domain.
The wrapper may mutate only that block job's private launch configuration,
which is not observable by a kernel or another worker.

The lowering may use a dynamic wrapper loop, unroll only a statically bounded
call shell, or inline one packet body into one dynamic loop. The handwritten
lowering must not clone the source body once per packet; LLVM may subsequently
inline a sufficiently small internal direct-CFG body under its ordinary target
cost model, while large scheduled bodies remain one local function. The
original packet function has internal linkage in batch mode, and exactly one of
the ordinary entry and batch entry is externally discoverable. W1, a
single-packet block, the disabled oracle, and standalone lowering retain the
ordinary entry. W8 inlining is a measured target policy and requires
TargetTransformInfo evidence for a wide register file. W16 inlining additionally
requires the linear-1D narrowing path, exactly one Schedule block, and 8--32
Schedule instructions; it is not enabled for a mixed scheduler CFG.
Correctness does not depend on either policy.
`LUISA_SIMD_DISABLE_W16_LINEAR_1D_PACKET_INLINE=1` restores the bounded W16
call shell. `LUISA_SIMD_DISABLE_LINEAR_1D_PACKET_TAIL_NARROWING=1` restores
per-packet dispatch checks, and `LUISA_SIMD_DISABLE_LINEAR_1D_THREAD_ID=1`
restores power-of-two thread-coordinate decomposition. Standalone packet
lowering always retains that general decomposition because its caller may
supply an arbitrary thread origin. `LUISA_SIMD_DISABLE_PACKET_BATCH_ENTRY=1`
restores the host-side single-packet loop before JIT construction.

For a direct-CFG runtime kernel, a thread-pool chunk containing consecutive
flattened blocks may instead enter an exclusive block-range wrapper. The first
three physical parameters are unchanged; the fourth is the number of blocks,
not the number of packets. The private launch configuration supplies the
three-dimensional grid size. Starting at its authored `block_id`, the wrapper
must visit exactly that many blocks in x-major order, wrap x then y, reset
`thread_index` to zero for every block, and issue the statically known complete
packet count through the internal block-local wrapper. A zero block count is a
no-op. The ordinary packet body remains responsible for active-tail and exact
dispatch-extent masking before any potentially trapping, poison-producing, or
memory operation.

If the range is also a linear 1D dispatch, the implementation may replace the
per-block loop by one packet range only after proving all of the following:
direct control flow; one Schedule block with at most 32 instructions; an
inlined, statically bounded packet body; no alloca or block barrier; no
`thread_id` or `block_id`; and every direct `dispatch_id` use is an in-range
constant extraction of component zero. The wrapper dynamically rechecks
`dispatch_size.y == dispatch_size.z == 1` and otherwise takes the generic
x-major block loop. The coalesced packet index may exceed the authored block's
local x range only because the proof excludes every observation of local
thread/block identity; `block_id.x * block_size.x + packet_index` remains the
same global `dispatch_id.x`. Packet boundaries are unchanged, the final tail
is narrowed first, and the wrapper restores the generic path's final
`block_id` and `thread_index`. `LUISA_SIMD_DISABLE_LINEAR_1D_BLOCK_COALESCING=1`
is the differential oracle.

The block-range entry is legal only when direct control flow, a static nonzero
packet count, and the packet-batch entry are all available. Scheduler-backed
kernels must export the block-local packet wrapper even when block-range
batching was requested; they may not acquire a second outer scheduler loop.
Exactly one of the ordinary, packet-batch, and block-range symbols is returned
by the JIT facade. `LUISA_SIMD_DISABLE_BLOCK_BATCH_ENTRY=1` selects the
block-local oracle before compilation.

The packet argument record, return-lane storage, and launch configuration are
separate ABI objects. The packet body only reads the argument record and launch
configuration, and the runtime always supplies a nonnull launch configuration.
LLVM may therefore attach `noalias readonly` to the body's argument record and
`noalias nonnull readonly` to its launch configuration. Packet/block wrappers
retain `noalias readonly` on the argument record and `noalias nonnull` on the
launch configuration, but must not claim the latter is `readonly`: they store
the current packet/block indices. These facts say nothing about aliasing among
resource addresses loaded from the record. Callers must not overlap return-lane
storage with the argument record. The diagnostic
`LUISA_SIMD_DISABLE_PACKET_ABI_ALIAS_ATTRIBUTES=1` removes these attributes
without changing the physical ABI or execution result.

Each block job owns its `SIMDPacketLaunchConfig`; the argument descriptor buffer
is immutable for the duration of the dispatch. Dispatch-edge packet masks are
therefore unchanged by host scheduling. Concurrent non-atomic conflicting
access from different blocks remains a source-level data race, while existing
LLVM atomic lowering supplies the required cross-worker ordering. Shader
submissions sharing one device pool are serialized to protect pool state; no
additional ordering between independent host streams is implied.

`SIMDDeviceConfigExt::worker_count()` defines the host execution width. Zero
selects `max(hardware_concurrency, 1)` and one executes block ranges inline for
diagnostics and serial differential benchmarks.

Reproducible example sweeps may use `LUISA_SIMD_WARP_WIDTH=1|2|4|8|16` when
the application does not construct `SIMDDeviceConfigExt`; an explicit nonzero
API width always wins. `LUISA_SIMD_DISABLE_PREDICATED_IF=1` and
`LUISA_SIMD_DISABLE_LOOP_UNSWITCH=1` provide control-flow A/B controls;
`LUISA_SIMD_DISABLE_GUARDED_LOOP_UNSWITCH=1` keeps constant-trip unswitching
but rejects an unknown-trip entry guard;
`LUISA_SIMD_DISABLE_PREDICATED_LOOP=1` restores the generic loop scheduler,
while `LUISA_SIMD_FORCE_PREDICATED_LOOP=1` is a diagnostic profitability
override that retains all semantic safety checks;
`LUISA_SIMD_DISABLE_PREDICATED_IF_REFINEMENT=1` disables nested select/Phi
forwarding, while
`LUISA_SIMD_DISABLE_DEEP_PREDICATED_IF_REFINEMENT=1` keeps forwarding but
restores the W8 speculation-cost ceiling from sixteen to twelve;
`LUISA_SIMD_DISABLE_WIDENED_PREDICATED_UPDATE=1` disables the separately
costed one-sided update policy at W2/W4/W8/W16;
`LUISA_SIMD_DISABLE_RAY_QUERY_FILTER_PREDICATION=1` restores the scheduled
cutout-filter diamonds, while
`LUISA_SIMD_FORCE_RAY_QUERY_FILTER_PREDICATION=1` bypasses only their W8
profitability gate for semantic tests;
`LUISA_SIMD_DISABLE_COHERENT_MASK_REUSE=1` restores derived successor masks on
runtime-coherent varying branches and switches;
`LUISA_SIMD_DISABLE_DIRECT_DIVERGENT_CHILD=1` restores the explicit LIFO
push/pop for the selected divergent child, while
`LUISA_SIMD_FORCE_DIRECT_DIVERGENT_CHILD=1` is its low-state diagnostic force;
`LUISA_SIMD_DISABLE_STATE_PHI_COALESCING=1` restores one physical slot per
logical scheduled PHI version;
`LUISA_SIMD_DISABLE_COHERENT_DIRECT_CFG=1` forces otherwise coherent functions
through the general cohort scheduler;
`LUISA_SIMD_DISABLE_AGGREGATE_PROMOTION=1` restores aggregate local storage
before the two SROA/`mem2reg` stages;
`LUISA_SIMD_DISABLE_UNIFORM_BUFFER_BROADCAST=1` controls the typed-buffer
refinement, and `LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER=1` controls proven
lane-consecutive typed-buffer accesses.
`LUISA_SIMD_DISABLE_PAIRED_LEAF_GATHER=1` restores separate 32-bit leaves for
eligible W8 direct vector-buffer reads. `LUISA_SIMD_REPORT_OPTIMIZATIONS=1`
logs per-shader transform, scheduler-state, ray-query scratch, ray-query
status-color, and cached state-handle counters.
`LUISA_SIMD_DISABLE_PACKET_BATCH_ENTRY=1` restores one exported JIT call per
packet and is the runtime/codegen A/B oracle for block-local packet batching.
`LUISA_SIMD_DISABLE_BLOCK_BATCH_ENTRY=1` restores one block-local packet-batch
call per block for otherwise eligible direct-CFG kernels.
`LUISA_SIMD_DISABLE_UNIT_DIMENSION_MASK_ELISION=1` restores dispatch compares
for statically unit block dimensions.
`LUISA_SIMD_DISABLE_LINEAR_1D_THREAD_ID=1` restores general power-of-two
thread-coordinate decomposition in runtime 1D packets.
`LUISA_SIMD_DISABLE_LINEAR_1D_PACKET_TAIL_NARROWING=1` restores per-packet
dispatch masking instead of one full-width loop plus one prefix tail.
`LUISA_SIMD_DISABLE_W16_LINEAR_1D_PACKET_INLINE=1` retains the W16 packet call
shell for the otherwise eligible small straight-line 1D body.
`LUISA_SIMD_DISABLE_LINEAR_1D_BLOCK_COALESCING=1` restores the generic
per-block wrapper for a proven block-agnostic linear range.
`LUISA_SIMD_DISABLE_PACKET_ABI_ALIAS_ATTRIBUTES=1` removes the read-only,
nonnull, and no-alias packet ABI facts used to hoist launch and descriptor
loads; it does not alter resource aliasing semantics.
The optimization report exposes these launch refinements as
`unit_dimension_mask_elisions`, `linear_1d_thread_ids`,
`linear_1d_packet_tail_narrowings`, and `linear_1d_block_coalescings`.
`LUISA_SIMD_REPORT_XIR=1` logs canonical XIR immediately before and after the
SIMD scheduling rewrites. `LUISA_SIMD_REPORT_SCHEDULE=1` logs the verified
Schedule IR immediately before LLVM lowering, including value classes,
convergence/loop metadata, resource access annotations, edge assignments, and
terminators. It is an explicit diagnostic because large graphics kernels
produce correspondingly large logs.
`LUISA_SIMD_REPORT_ASSEMBLY=1` additionally captures optimized target assembly
and reports its static instruction/call/branch counts, stack references, the
x86 stack allocation when recognizable, and scalar-math symbols.
`LUISA_SIMD_DUMP_ASSEMBLY_DIR=<directory>` writes matching, uniquely named
`.s` and `.o` files. The annotated `.s` is compiled with the same explicit
PIC/small target model as ORC and supplies LLVM basic-block labels; the `.o`
is the exact relocatable compiler output consumed by ORC before JITLink and is
authoritative for object bytes, symbols, and disassembly. The two artifacts
have matching instruction and basic-block offsets, although assembler padding
encodings need not be byte-identical. `LUISA_SIMD_REPORT_JIT_ADDRESS=1` logs
the live entry address so profiler records can be correlated with the object
symbol/section offsets. `LUISA_SIMD_DISABLE_COLD_STATE_PARTITION=1` and
`LUISA_SIMD_DISABLE_RAY_QUERY_SCRATCH_COLORING=1` are same-binary A/B controls
for cold-state pinning and ray-query scratch coloring. Cold-state access counts
are over distinct physical slots; a state set compacted by PHI coalescing stays
promotable because the measured volatile-pinning variant is slower.
`LUISA_SIMD_DISABLE_RAY_QUERY_STATUS_CACHE=1` restores authoritative AoS
predicate gathers for eligible ray queries.
`LUISA_SIMD_DISABLE_RAY_QUERY_STATE_HANDLE_CACHE=1` keeps the packed status
sidecar but restores the query-local pointer gathers.
`LUISA_SIMD_DISABLE_RAY_QUERY_STATUS_CALLBACK_PAIRING=1` keeps both caches but
restores the JIT-side plain-callback gather and cohort check.
`LUISA_SIMD_DISABLE_PROCEDURAL_WIDE_STATUS_PACK=1` disables W16 procedural
status specialization entirely;
`LUISA_SIMD_DISABLE_PROCEDURAL_WIDE_FUSED_STATUS=1` retains that specialization
but restores its preceding plain-provider-plus-pack implementation.
These process-wide variables are diagnostic controls, not shader semantics or
a replacement for the public configuration extension. Invalid width text or a
width outside the supported set is rejected.

### 4.3 Static power-of-two block specialization

The production compiler forwards a kernel's declared three-dimensional block
size into Schedule-to-LLVM. Every nonzero static dimension must be a power of
two. Static `thread_id` decomposition then uses masks and logical shifts; it
must not emit integer division or remainder. The generic public lowering API
may pass zeros to request the runtime launch-config path, which retains
`udiv`/`urem` because no static divisor is available.

This is a compile-time contract, not an unchecked optimization hint. A kernel
with a non-power-of-two static dimension is rejected before LLVM emission.
Regression IR verifies `{32, 2, 1}` has named mask/shift decomposition and no
`udiv`/`urem`, while `{48, 2, 1}` fails with a precise diagnostic.

### 4.4 Predicated diamond refinement

A raw conditional diamond may be replaced by straight-line predicated code
only when the source branch condition is classified `varying`, both arms are
speculation-safe, structural and weighted-cost limits pass, and every merge
PHI has one well-typed incoming value from each arm. Warp- or cohort-uniform
conditions remain scalar branches. W1 is never if-converted: its direct scalar
CFG cannot diverge, so executing both arms has no scheduling benefit. The
transformation is a refinement of the same per-lane CFG: for lane `i`, the
generated select chooses exactly the PHI incoming associated with the scalar
edge selected by that lane.

Operand sanitization remains a precondition, not a post-hoc mask. Operations
that may trap, form poison, index an invalid aggregate element, access memory,
or produce a side effect are excluded even if their result would later be
selected away. Inactive dispatch-tail lanes therefore execute only total pure
operations before their results are discarded by the active mask.

Cast safety is classified by operation and source/target type. Bitwise casts
are total. A verifier-valid static cast is accepted unless it converts a
floating-point scalar/vector to a signed or unsigned integer scalar/vector;
LLVM permits that conversion to produce poison for NaN and out-of-range input.
Integer-to-float, integer-width, float-width, Boolean, and float-to-Boolean
conversions are total under the backend lowering and may be hoisted within the
normal cost limits. The permanent pass regression keeps float-to-integer
rejected, while Schedule codegen executes an integer-to-float diamond at W8
with a 13-element inactive tail.

The follow-up identity

```text
select(f(a), f(b), c) = f(select(a, b, c))
```

is legal only for matching pure total operations. With a vector condition,
`f` must be component-wise; with any condition, the selected operand types
must match, exactly one operand pair may differ, both producers must be
single-use, and all affected instructions must be free of local metadata.
This rewrite reduces already-speculated work; it does not authorize a diamond
that failed the original safety test or introduce a fast-math domain
extension. IR-shape, inactive-tail execution, final assembly, and throughput
are permanent gates in `benchmark_simd_predicated_if` and the Schedule-codegen
regressions.

At W4/W8 only, an additional bounded refinement may expose the next enclosing
diamond in a nested select ladder. It may bypass a single-predecessor PHI-only
forwarding block only when at least one incoming value is a select generated
by the immediately preceding if-conversion round. Every PHI must have exactly
one well-typed incoming from that predecessor; the block and branch must have
no metadata; the downstream target must have exactly one corresponding PHI
edge, at least one sibling predecessor, and no already-existing edge from the
predecessor. A PHI may carry only Name metadata, which moves to its unique
select value or must already match that value's Name. Any non-Name metadata,
conflicting Name, multiple use, pre-existing select, or ambiguous edge rejects
the complete forwarding block.
The original speculation-safety and cost contract is then reapplied to the
newly exposed diamond. The transform is capped at eight conversion rounds and
eight forwarding blocks per round. A loop-carried PHI has multiple incoming
edges and is rejected by the single-incoming structural requirement.

This restriction is a performance policy, not a semantic width limitation.
Default-worker Voxel A/B accepts W4/W8 and rejects W2 (regression) and W16
(neutral); W1 never enters varying if-conversion. Thirteen-element inactive-
tail execution, same-binary oracle comparison, non-Name metadata rejection,
and pre-existing-select provenance are permanent regressions.

One further cost boundary is enabled only at W8. After the ordinary
select/Phi refinement has exposed another enclosing diamond, the weighted
speculation ceiling may rise from twelve to sixteen register units. All
structural, totality, metadata, live-out, per-arm, and six-instruction gates
remain unchanged; the higher budget does not admit memory, division, poison-
forming casts, or side effects. It exists for the measured four-register-unit
`float3` select-ladder shape whose next layer costs fourteen units.
`LUISA_SIMD_DISABLE_DEEP_PREDICATED_IF_REFINEMENT=1` restores the cost-twelve
policy without disabling the already accepted W4/W8 forwarding refinement.
W4 keeps cost twelve because a fourteen-pair Voxel A/B measured a regression;
W1/W2/W16 are byte-identical with the control. A W2/W4/W8/W16 `float3` ladder,
thirteen-element inactive tail, exact oracle equality, Schedule-state
reduction, and final x86 assembly-size reduction are permanent regressions.

After that W8 deep refinement, one further conversion is legal only for the
measured material-ladder shape. One arm is empty; the other contains exactly
three scalar Boolean `BINARY_EQUAL` operations and three float32x3 `SELECT`
operations. Both unannotated arm exits target the same merge, and exactly one
well-typed float32x3 PHI differs between the arms. The generic pass still
proves single predecessors, pure total instructions, no poison-forming cast,
no memory/call/effect, no ambiguous metadata owner, at most six total
instructions, four live-out register units, and weighted cost no greater than
nineteen. This separate pass does not raise the ordinary cost-sixteen ceiling
and cannot admit an unrelated cheaper-arm shape.

Only W8 enables the conversion. The same-binary oracle is
`LUISA_SIMD_DISABLE_WIDE_PREDICATED_IF_REFINEMENT=1`; the optimization report
counter is `predicated_wide_select_ladder_diamonds`. Permanent execution uses
a thirteen-element inactive tail and exact oracle comparison. It requires one
W8 conversion with six hoisted instructions and smaller Schedule/assembly,
while W2/W4/W16 must retain byte-identical assembly. The rule neither reads an
inactive address nor extends the domain of an arithmetic operation.

One-sided state updates have a separate measured policy. It accepts only a
varying diamond with one empty arm, five or six non-terminator instructions in
the other arm, one direct common merge, at least two differing merge PHIs, no
more than six 32-bit live-out units, and a weighted speculation cost no greater
than fifty-eight. Floating division has latency weight eight, matching
component-wise matrix division, rather than the default arithmetic weight.
The default pure/total instruction and metadata rules continue to apply. The
only additional operation is scalar or vector floating-point division. It is
non-trapping on supported XIR targets: zero, signed zero, NaN, Inf, and
subnormal operands retain the backend's ordinary floating semantics, and the
select prevents any formerly untaken result from becoming observable.
Floating exception flags are not part of the Luisa device-language contract.
Integer division, floating/integer remainder, shifts, dynamic indexing,
float-to-integer conversion, memory, calls, and side effects remain rejected.
This opt-in does not change the target-independent if-conversion default and
does not extend the domain of an observable operation.

W2/W4/W8/W16 enable the policy; W1 and
`LUISA_SIMD_DISABLE_WIDENED_PREDICATED_UPDATE=1` retain the original CFG. A
permanent regression compiles both forms at every width, checks the widened
counter and exact Schedule-state reduction, executes a thirteen-element
inactive tail, compares every output bit, and requires smaller final x86
assembly at each enabled width. A separate XIR regression proves that floating
division requires the explicit option and integer division remains
ineligible. The execution fixture also speculates a zero denominator in a
formerly untaken lane and proves that its selected output is unchanged.

#### 4.4.1 Static-extract ray-query filter refinement

The generic XIR if-conversion safety set may include `EXTRACT` only when its
caller explicitly enables `allow_speculative_static_extract`. The pass decodes
every index as a nonnegative integer constant, checks it at each array, vector,
matrix, or structure level, and requires the final walked type to equal the
instruction result type. Failure at any level rejects the diamond atomically.
The option remains false for every ordinary target-independent pipeline. A
dynamic extraction is never accepted, and proving an extraction total does not
authorize speculation of its aggregate producer.

The SIMD ray-query policy is narrower still. Its condition is varying and must
be `SurfaceHit::inst == constant` (in either operand order), where member zero
is extracted directly from one
`RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT` read. Both unannotated plain arms have
one direct common merge and exactly the measured `(0, 5)` or `(5, 8)`
nonterminator counts. Arm instructions may contain only bounded static extracts
from that same hit, floating multiply, less/equal comparison, `fract`, and
select; at least one `fract` is present and exactly one Boolean PHI differs at
the merge. The generic predecessor, metadata, live-out, type, purity, and
speculation-cost checks are then reapplied. A different hit field/root, dynamic
index, memory operation, call, query operation, or side effect retains the
original CFG.

The candidate-hit read remains inside its original surface-handler execution
domain. Its masked gathers already supply zero to inactive physical lanes
before the newly hoisted extracts or arithmetic execute. For every active lane
whose original branch chose one filter arm, the final select observes exactly
that arm's Boolean result. Arithmetic from the other arm may produce ordinary
IEEE NaN/Inf values, but it cannot trap, form an invalid address, mutate query
state, or become observable. This does not extend the accepted input domain or
change `fract`, NaN, Inf, signed-zero, or subnormal semantics on the selected
path. Commit, terminate, and `proceed` ordering are unchanged.

Production enables the policy only at W8 and attempts at most three conversion
rounds. `LUISA_SIMD_DISABLE_RAY_QUERY_FILTER_PREDICATION=1` restores the
original scheduler path in the same binary.
`LUISA_SIMD_FORCE_RAY_QUERY_FILTER_PREDICATION=1` bypasses only the width gate;
all semantic checks remain mandatory. The optimization report counter is
`predicated_ray_query_filter_diamonds`. Permanent execution compiles candidate
and oracle at W1/W2/W4/W8/W16, forces nonproduction widths, executes a
13-element dispatch with inactive tails, supplies tall/short/other candidates
through both plain and packed-status query ABIs, and requires bit-identical
outputs and untouched inactive sentinels. A `SurfaceHit::prim` lookalike and a
dynamic XIR extraction fail closed.

#### 4.4.2 Predicated direct-memory diamond

Schedule-to-LLVM may also straighten a small varying diamond containing direct
typed-buffer reads. This is separate from the XIR transformation above: memory
is not speculated under the outer packet mask. The emitter computes
`A_true = A & condition` and `A_false = A & !condition`, executes each arm
under its own mask, applies its edge assignments with that same mask, restores
`A`, and continues at the common merge. A masked-off lane therefore neither
loads memory nor updates a PHI/state destination.

The current candidate is fail-closed and requires all of the following:

- W2/W4/W8/W16, a varying conditional split, two distinct single-predecessor
  arm blocks, one common merge, exact convergence joins, and no loop back;
- no split-edge assignments and only varying/mask destinations on arm exits;
- at most eight arm instructions in total and at least one nonvolatile direct
  typed `BUFFER_READ` with varying index and result;
- only total arithmetic, comparisons, selects, bit operations, and safe casts
  around those reads. Division/remainder, float-to-integer conversion, calls,
  stores, atomics, byte/bindless/texture/accel access, and every other effect
  remain rejected.

The direct buffer implementation forms a non-`inbounds` address and issues the
load with the arm mask before any result is observed. For an arm-local or
non-affine index, inactive elements are selected to zero before either a
dynamic seed extraction or gather-pointer formation. An empty arm therefore
forms only a benign masked-off address and touches no memory. If a proven
lane-consecutive index was defined outside the arm, its conceptual packet base
may be reconstructed from the outer packet's safe seed; this avoids a
horizontal extract and spill caused only by the submask. An arm-local index
must instead use the arm's safe first lane because its inactive elements may
be poison. W1 retains the ordinary scalar branch.

`LUISA_SIMD_DISABLE_PREDICATED_IF=1` disables both predication families for
differential diagnostics. `LUISA_SIMD_DISABLE_PREDICATED_IF_REFINEMENT=1`
retains the first small-diamond pass but disables only the W4/W8 forwarding
refinement. Permanent coverage includes an underflowing index
in the untaken lane, a completely empty arm with a null input pointer,
W2/W4/W8/W16 inactive tails, nesting under an outer scheduled convergence,
volatile and division rejection, the W1 path, optimized assembly with no stack
frame, and runtime counters for accepted diamonds/instructions.

### 4.4.2 Full-packet coherent-region versioning

Schedule-to-LLVM may duplicate one short successor region when a runtime-
coherent varying conditional is also a physically full packet. Let `A` be the
nonempty incoming mask and `T = A & C`, `F = A & !C`. The existing coherent
path proves exactly one of `T` and `F` is nonempty and therefore that its
selected successor mask equals `A`. The additional guard is

```text
all_on = and.reduce(A)
```

If `all_on` is true, `A` is the all-one mask and the selected region may use a
constant all-one mask plus seed lane zero. If it is false, including every
partial tail, the original scheduler edge executes. The genuinely divergent
path does not evaluate this guard and retains its convergence frame and
worklist records.

An eligible region must satisfy all of the following:

- W2 or W8; at most one region per function, chosen in Schedule source order;
- one arm of a varying conditional with a declared convergence, a clean
  non-loop entry edge, and exactly one static convergence point targeting its
  merge;
- an acyclic, nonrepeating chain that joins exactly that convergence once and
  ends at the next varying conditional after at most four blocks;
- no foreign convergence target, loop back, memory, call, side effect, or
  terminal predicated-memory diamond;
- only the audited arithmetic/static-cast/bit-cast whitelist, no more than
  twenty-four weighted 32-bit register units;
- at least three blocks at W8. The structurally minimal two-block form remains
  legal only at W2.

If both arms qualify, codegen clones only the lower-cost one and breaks a tie
toward the false/miss arm. The cloned region routes every original edge and
assignment under the proven all-one mask. The coherent root did not allocate
the conditional's convergence frame; the unique-target condition therefore
makes its join gate an identity and prevents accidentally skipping a foreign
same-target frame. The final varying split uses normal scheduler lowering with
recursive region versioning disabled.

This transformation does not speculate an instruction: the other arm never
executes, every physical lane in the clone was active in the source cohort,
and no operation crosses its controlling branch. Consequently it neither
extends an arithmetic domain nor creates an inactive-lane memory/poison
observation. All rejected, partial-tail, and mixed cases preserve the original
independent-PC path.

`LUISA_SIMD_DISABLE_ALL_ON_REGION_VERSIONING=1` is the differential oracle.
`LUISA_SIMD_REPORT_OPTIMIZATIONS=1` reports accepted region, block, and
instruction counts. Permanent W2/W4/W8/W16 codegen coverage executes coherent
and genuinely divergent inputs with full, partial-tail, and one-lane masks;
it also fixes the W8 three-block minimum and W2 two-block exception.

### 4.4.3 Innermost-loop local predicated regions

This refinement removes a bounded amount of independent-PC scheduling inside
an innermost natural loop without requiring the complete loop to satisfy the
predicated-batch contract. It does not alter Schedule IR, clone a source
instruction, or change the fallback path for an unrecognized shape.

For an incoming active mask `A` and a varying Boolean condition `C`, define

```text
T = A & C
F = A & !C
```

A base local diamond is eligible only when all of the following hold:

- W2/W4/W8/W16; the split, both arms, and common merge are members of the same
  innermost natural loop;
- the split declares one convergence point, has distinct targets, no edge
  assignments/joins/loop backs, and its condition is varying;
- each arm is a one-to-three-block single-predecessor chain ending at that
  exact convergence, with no intermediate join or loop back;
- every arm-exit assignment destination is varying or a lane mask, and exactly
  one arm contains zero instructions;
- either the complete diamond is assignment-only, or the nonempty arm contains
  4--24 pure instructions including at least one `sqrt`, `rsqrt`, or floating-
  point division;
- the remaining accepted instructions are the audited nontrapping arithmetic,
  comparison, select, `min`/`max`, bit operation, or aggregate-construction
  set. Calls, memory/resources, effects, participant masks, integer
  division/remainder, dynamic extraction, and structured control fail closed.

A two-sided local diamond is a separately costed extension. At W2/W4/W8, both
arms may contain instructions when the structural conditions above still hold
except for the one-empty-arm and expensive-operation requirements. Both arm
counts are nonzero, and their total is 4--24. Its whitelist adds static and
bitwise casts to the same pure arithmetic set. W16 does not enable this form in
production; the diagnostic force knob retains identical semantic checks. No
arm, merge, or loop block is cloned.

Every instruction-bearing arm is emitted behind `any(T)` or `any(F)`. If that
mask is empty, the arm is not evaluated. If it is nonempty, fixed-vector
arithmetic may still evaluate inactive physical elements. Floating division by
zero and square root of a negative off-arm operand may produce the ordinary
IEEE Inf/NaN value; neither traps nor forms poison, and floating exception flags
are not part of the device-language contract. For a two-sided static
floating-to-integer cast, codegen selects zero into every off-arm lane before
forming `fptosi`/`fptoui`; masking the result afterwards is forbidden. Active
lanes retain the source-language domain unchanged. Integer division, memory,
effects, and every other poison/trap-capable operation remain excluded. This
permits no undefined domain extension. An assignment-only diamond emits masked
assignments directly and needs no horizontal mask guard.

After both arms, codegen restores `A` and the enclosing cohort's seed lane and
continues at the declared merge. Arm blocks are assigned to the split's LLVM
emission region during spill analysis. A value used only within the inlined
region consequently remains LLVM SSA rather than acquiring a Schedule spill;
cross-region values retain the existing spill/liveness rules.

One nested shape is legal at every enabled width. Exactly one outer arm must
contain a 1--12-instruction pure block ending in an assignment-only base
diamond; the sibling arm is empty. The inner convergence must name the outer
convergence as its parent, the inner merge and sibling must close the outer
convergence directly, and all involved blocks must have the exact predecessor
counts implied by this tree. The outer nonempty mask is tested before its
instructions or inner selector execute. Arbitrary recursive nesting is not
accepted.

At W4/W8/W16, a chained region may consume up to four adjacent nonempty base
diamonds. A transition starts at the previous diamond's merge only when that
block is the exclusive target of the expected convergence. It may cross no
more than four blocks and twelve instructions; every later bridge block has
one predecessor, is not a convergence target, stays in the same innermost
loop, and contains only audited pure arithmetic/casts. No bridge may loop
back, join, self-cycle, call, read memory, or produce an effect. The complete
chain is capped at 128 instructions. W8 may end with one nested shape satisfying
the preceding proof; W4 and W16 do not absorb that tail. No source instruction
is duplicated: chaining only makes the previous merge, pure bridge, and next
split one LLVM emission region and performs one final return to the scheduler.

The last local diamond may additionally absorb a terminal merge tail. Its
first block must be the exclusive target of that diamond's exact convergence,
and every later block must have one predecessor, remain in the same innermost
loop, and not be any convergence target. The region contains at most four such
blocks and 96 instructions and stops before a separately recognizable local
diamond, nested region, or predicated-memory diamond. Its final terminator is
emitted by the ordinary complete scheduler lowering.

This tail has a stronger execution identity than speculative if-conversion:
both arms have already completed, so codegen restores the original outer mask,
seed lane, and local SSA environment before executing the merge. Every tail
instruction, memory operation, resource access, call, and effect therefore
observes the same participant cohort and program order as the source; no
inactive operand becomes newly evaluated. Inlined blocks have no other static
predecessor and are removed from the dispatcher, so no instruction is cloned
or executed twice. Failure of any predecessor/convergence/loop/bound proof
leaves the merge on the original scheduler path.

Width policy is empirical but semantics are width-independent. W2 retains
individual diamonds because chaining regressed its paired benchmark. W4/W8/W16
enable ordinary chaining. Only W8 enables chained nested-tail absorption after
paired W4/W16 ablations measured small regressions. Terminal bridges are
production-enabled at W8/W16; W4 additionally requires at least 32 terminal
instructions, and W2 is available only through a diagnostic force override.
Two-sided local diamonds are enabled at W2/W4/W8; W16 retains its independently
profitable predicated-loop path after the forced local candidate was neutral.
The complete disabled
oracle is `LUISA_SIMD_DISABLE_LOCAL_PREDICATED_REGIONS=1`; narrower oracles are
`LUISA_SIMD_DISABLE_LOCAL_PREDICATED_CHAINING=1`,
`LUISA_SIMD_DISABLE_NESTED_PREDICATED_REGION=1`, and
`LUISA_SIMD_DISABLE_CHAINED_NESTED_TAIL=1`. The terminal differential oracle
is `LUISA_SIMD_DISABLE_LOCAL_PREDICATED_TERMINAL_BRIDGE=1`; the diagnostic-only
W2 override is `LUISA_SIMD_FORCE_LOCAL_PREDICATED_TERMINAL_BRIDGE=1`.
`LUISA_SIMD_DISABLE_TWO_SIDED_LOCAL_PREDICATION=1` isolates the two-sided form,
while `LUISA_SIMD_FORCE_TWO_SIDED_LOCAL_PREDICATION=1` exercises W16.

Permanent codegen coverage executes candidate and every narrower oracle at
W2/W4/W8/W16 with `W - 1` active lanes. It includes a guarded square-root arm,
a guarded floating-division/`max` arm with no square root, an assignment-only
nested tail, and an independently varying natural loop. Every output bit must
match and every inactive output element must retain its sentinel. Runtime
counters separately report local diamonds, assignment-only diamonds, blocks,
instructions, nested regions, chained regions, transitions, blocks, and
chained nested tails. A separate terminal-bridge regression executes the
forced W2 and production W4/W8/W16 paths with `W - 1` active lanes, a guarded
square root, a 41-instruction merge-to-loop-back tail, LLVM verification,
machine-code scalar-libm rejection, and exact disabled-oracle equality.
Runtime counters additionally report terminal blocks and instructions.
A separate two-sided regression covers production W2/W4/W8 and forced W16,
`W - 1` active lanes, independently varying natural-loop iterations, nonempty
arms, an off-arm NaN before `fptoui`, operand pre-sanitization, terminal merge
absorption where enabled, LLVM verification, and exact disabled-oracle bits.
The optimization report exposes its count independently from the older
one-sided and assignment-only families.

### 4.5 Bounded loop-unswitch refinement

The production SIMD compiler may replace a repeated internal conditional with
one conditional before two cloned loop versions. This is permitted only when
all of the following hold:

- the destructured natural loop is innermost and has one preheader, latch,
  exit edge, and exit block. A constant trip count must be greater than one;
  an unknown count requires the guarded top-tested form below;
- the complete loop has at most 48 instructions and the compiler transforms at
  most one loop per function;
- the selected conditional dominates the latch, both targets remain inside the
  loop and are not the header, and its nonconstant, non-`undef` Boolean
  selector is defined outside the loop;
- the SIMD uniformity analysis classifies the selector exactly `varying`;
- every loop terminator is a plain conditional/unconditional branch and the
  body contains no clock, volatile instruction, write, call, collective, or
  other operation modeled as cohort-sensitive;
- every direct live-out is an rvalue whose external ordinary uses are dominated
  by the unique exit. Existing exit PHIs must have exactly one incoming value
  from the original exit edge.

The original loop is specialized to the true edge and the clone to the false
edge. For a proven positive count, the old preheader dispatches to two
canonical preheaders. Existing exit PHIs receive the cloned incoming edge;
direct live-outs receive one new exit PHI. Candidate branch metadata moves to
the outer dispatch, and structured CFG causes atomic rejection before
mutation. Rejecting writes and clocks prevents the cohort-order change from
becoming observable.

For an unknown count, guarded unswitching is permitted only when the unique
exit is selected by the top-tested header branch. A new entry guard clones the
header condition with every header PHI replaced by its preheader incoming
value. Only arithmetic in the condition or required zero-trip exit/live-out
values may be cloned. A lane whose guard is false goes directly to the source
exit with those resolved values; therefore it never evaluates the invariant
selector. Entering lanes reach the two-version dispatch. All new exit-PHI and
live-out incoming values must be materializable by the same resolver before
mutation. `LUISA_SIMD_DISABLE_GUARDED_LOOP_UNSWITCH=1` rejects only this form.

The outer selector still follows the ordinary scheduler rule. If its active
values happen to agree at runtime, the dynamically coherent fast path directly
enters one specialized loop; otherwise the packet splits once and each cohort
executes its selected version. Inactive tail lanes remain outside both masks.
The rewrite introduces no speculative arm evaluation and does not relax the
operand-sanitization requirement.

Correctness gates cover cloning, cyclic PHIs, exit-PHI and direct-live-out
repair, metadata, structured-module atomicity, guarded zero/mixed/positive
dynamic trip counts, the guarded-disabled oracle, `undef`, clock/write
rejection, all supported SIMD widths, and inactive tails.
`benchmark_simd_loop_unswitch` additionally audits optimized assembly, calls,
stack references, and repeated throughput.

### 4.5.1 Bounded predicated innermost-loop batch

This LLVM refinement is independent of XIR loop unswitching and does not clone
the loop body. Schedule IR may annotate a natural loop with
`max_trip_count=N` only when XIR analysis proves that no execution can perform
more than `N` body iterations. Extra early exits may reduce the actual count;
they cannot extend it. For a top-tested loop, the batch executes at most
`N + 1` header evaluations so the final false test is included.

The default candidate must satisfy every condition below:

- it is the only selected loop in the function, is innermost, has a varying
  header split, 6--24 blocks, at most 96 instructions, and
  `1 <= N <= 4096`;
- deleting its annotated backedges leaves a single-entry acyclic region whose
  only external entry is the header and whose escapes are declared loop exits;
- every flattened join is declared by a split in the same region; the header's
  convergence target is one of the declared exits;
- every result and edge-assignment destination is varying or a lane mask;
- instructions are restricted to the audited nontrapping arithmetic/select/
  compare set, static or bit casts, and direct nonvolatile typed-buffer reads
  with a varying index;
- nested loops, local-pointer operations, division/remainder/shifts, volatile,
  bindless or texture access, writes, atomics, calls, acceleration queries,
  barriers, returns, and collectives reject the complete candidate.

For each batch iteration, codegen computes one mask per region block in
topological order, applies assignments under that mask, accumulates each exit
mask, and ORs audited backedges into the next-iteration mask. A zero block mask
must never make an address or poison-producing conversion observable: resource
indices are selected to zero before a masked gather, and inactive
floating-point operands are selected to zero before `fptosi`/`fptoui`. The
latter protection also applies to ordinary varying Schedule blocks.

At batch exit, let the possible destinations be the loop header and every
declared exit whose accumulated mask is nonempty. Zero destinations is
unreachable for a live cohort. One destination continues without a new frame.
Two or more destinations declare exactly the original header convergence using
the union of all destination masks, queue each nonempty destination with that
token, and resume the general scheduler. A matching frame already at the top
is reused, retaining its original expected mask. This rule is required even
when no lane continues but two different early exits are nonempty.

Default profitability additionally requires LLVM TTI to report at least a
512-bit fixed-vector register and a legal non-scalarized masked gather. W16 is
then enabled for any device worker count. W8 requires at least 24 workers.
W1/W2/W4 and every target that fails the TTI gate retain the general scheduler.
`LUISA_SIMD_DISABLE_PREDICATED_LOOP=1` is the same-binary oracle;
`LUISA_SIMD_FORCE_PREDICATED_LOOP=1` bypasses only profitability gates for
diagnostics and semantic tests, never the structural or instruction-safety
checks. The optimization report exposes `dispatch_workers`,
`native_predicated_loop`, `predicated_loops`, block/instruction counts, and
`predicated_loop_batch_iterations`.

Permanent execution coverage includes an inactive W16 tail, finite-count and
sentinel early exits, multiple dynamic exit destinations, a post-loop warp
collective, inactive NaN lanes before floating-to-integer conversion, exact
candidate/oracle/scalar equality, W8 worker crossover selection, and rejection
of writes and volatile reads.

### 4.5.2 Structured early-exit innermost-loop refinement

This refinement is distinct from the topological predicated batch. It retains
the loop's structured backedge and one shrinking continuation mask, executes
audited early-exit tails when their masks become nonempty, and reaches one
common post-loop block after all lanes have left. It neither clones the loop
nor changes the Schedule scheduler's abstract result.

The production candidate is the first eligible W8 loop in Schedule order and
must satisfy all of the following:

- it is innermost, has no child loop, has 25--64 blocks and at most 256 loop
  instructions, and carries a proven `max_trip_count` in `[1, 16]`;
- its header is a split with the canonical
  `cohort_uniform_condition`, one declared convergence, and that convergence's
  target is a declared loop exit distinct from the header;
- the header predicate is arithmetic with exactly one varying state-slot
  operand and otherwise uniform operands. Only pure arithmetic/cast values
  transitively derived from that induction and uniform inputs may drive an
  internal split whose two targets both remain in the loop;
- every instruction produces a result, has no participant mask, and is pure
  arithmetic or a static/bitwise cast. Integer division, every remainder, and
  shifts are rejected; all memory/resource operations, local-pointer access,
  writes, atomics, calls, acceleration, collectives, barriers, and returns are
  consequently excluded;
- every nonheader loop block has at least one static predecessor and all of
  those predecessors are in the loop. Its remaining control is an audited
  branch/backedge, an inside/exit split, a cohort-equal inside/inside split, or
  an existing proven local predicated region;
- each declared exit reaches the header convergence's common target through a
  disjoint linear tail of at most four blocks and 64 instructions. A nonempty
  tail entry has exactly one predecessor from the loop, every later block has
  exactly its preceding tail block as sole predecessor, and every inside/exit
  split declares that same common target as its convergence.

The diagnostic force knob may lower the block minimum to four and bypass the
width gate; it bypasses none of the other clauses. Any failed clause leaves
every block on the original independent-PC path before IR emission begins.

Let `A0` be the nonempty mask that enters the header and `Ak` the continuation
mask at one iteration. For an exit condition `C`, codegen first sanitizes its
inactive lanes and forms the disjoint masks

```text
E   = Ak & C
Ak' = Ak & !C
```

with `C` inverted when the exit is the false edge. `E` applies the source edge
assignments and each tail assignment/instruction under `E`; `Ak'` applies the
continue edge and becomes the mask stored at the next backedge. An empty `E`
skips its tail. Because a lane leaves the loop exactly once, exit masks from
different edges or epochs are disjoint. Masked PHI/state writes therefore
preserve the value selected by that lane's actual exit and cannot overwrite a
different live lane.

The `max_trip_count` proof ensures that repeated backedges cannot leave a lane
in the loop forever. Once `Ak'` is empty, every lane in `A0` has written the
common-exit state. Codegen then enters that target once under `A0`, preserving
the source convergence ID and the current parent token. The ordinary
destination-side arrival cascade consequently observes the same complete
cohort and outer-frame state as the general scheduler.

Pure arithmetic may still be physically evaluated for inactive elements of an
executing fixed vector. Integer division/remainder/shifts are excluded even
though the general arithmetic emitter has its own sanitizer. A varying static
float-to-integer cast is permitted only because the structured emitter passes
the exact current mask to `_emit_instruction`; `_cast` selects zero into every
inactive leaf before forming `fptosi`/`fptoui`. Floating division is
nontrapping. Existing local predicated arms retain their own nonempty-mask
guards and pre-sanitization rules. No result mask applied after a poison-
forming operation is accepted as a substitute.

The accepted control-driving loop blocks and pure exit tails have no remaining
dispatcher entry. Their values can stay in LLVM SSA/registers, but this is not
a semantic requirement: target register allocation may still spill them. The
common exit, parent scheduler, resource ABI, packet width, and fallback backend
are unchanged. The implementation emits only target-independent fixed-vector
LLVM IR.

`LUISA_SIMD_DISABLE_STRUCTURED_EARLY_EXIT_LOOP=1` restores the general
scheduler in the same binary.
`LUISA_SIMD_FORCE_STRUCTURED_EARLY_EXIT_LOOP=1` is diagnostic/test-only.
`structured_early_exit_loops`, block/instruction counts, and absorbed-tail
blocks expose selection. Permanent coverage executes a forced 14-block W8
candidate and disabled oracle over a 13-thread dispatch, multiple exit epochs,
a two-sided local diamond, and inactive NaNs before `fptoui`; every active bit
and inactive sentinel must match the scalar reference. Separate resource-read
and integer-division variants, plus an otherwise eligible Schedule graph whose
inside/exit split names an in-loop convergence target, must report zero
candidates.

### 4.6 Cohort-equal typed-buffer read refinement

A nonvolatile typed `BUFFER_READ` may become one scalar load followed by a
fixed-vector splat when either its index value is globally warp/cohort-uniform
or Schedule lowering attaches a use-site cohort-equality proof. The current
use-site proof is deliberately narrow: a canonical integer induction PHI,
one preheader and latch, constant stride, uniform start, and a read whose block
is still inside that natural loop. Loop continuation identity separates
epochs at the read. The PHI itself remains lane-wise because an exit gate may
later merge different epochs.

The executing Schedule block always has a nonempty active mask. W1 therefore
uses lane zero; W4/W8/W16 extract the first active lane, load the complete
Luisa value once, and broadcast it. W2 retains the gather form for a
use-site-only proof because measured first-lane selection cost exceeds the one
saved lane load; a globally uniform scalar index still uses the scalar load.
Byte-address and volatile reads are excluded. No inactive address is formed,
and the transform does not authorize an out-of-domain or out-of-bounds access.

Permanent regressions cover global warp/cohort-uniform indices, a canonical
loop with nested varying control, W1/W2/W4/W8/W16, an inactive tail, volatile
reads, the disabled path, LLVM masked-gather shape, and final host assembly.
`LUISA_SIMD_DISABLE_UNIFORM_BUFFER_BROADCAST=1` provides the same-binary A/B
oracle, while `LUISA_SIMD_REPORT_OPTIMIZATIONS=1` reports the accepted read
count.

### 4.7 Paired 32-bit leaves in direct typed-buffer vectors

On a target where LLVM reports a 512-bit fixed-vector register and a legal,
non-scalarized `<8 x i64>` masked gather, W8 may combine each adjacent pair of
32-bit leaves in a direct, nonvolatile `BUFFER_READ` of a Luisa vector. One
masked 64-bit gather replaces two masked 32-bit gathers. Logical shifts,
truncation, and bitcasts reconstruct the two component-major vectors. The
little- and big-endian component orders are explicit; no target intrinsic is
emitted.

This is not a general aggregate rewrite. The current contract accepts only a
top-level vector with at least two non-Boolean 32-bit scalar elements. It does
not cross structure, array, matrix, bindless, byte-address, volatile, local,
acceleration-structure, or ray-query boundaries. Odd final vector elements
retain the ordinary 32-bit leaf gather. The exact active mask and the original
non-`inbounds` base offset are used for every pair, so inactive lanes form no
observable access and the optimization does not extend the source buffer
element beyond its declared layout.
The typed buffer's declared element must exactly match the read result, and
each paired pair must have adjacent four-byte field offsets; otherwise the
whole read fails closed to ordinary leaves.

W1/W2/W4/W16 and hosts without the required LLVM TargetTransformInfo proof
retain the ordinary leaf path. W4 was neutral and W16 was not stable enough in
real-render A/B, while W8 had a stable ordinary-path-tracing win on the audited
host. Target-unaware Schedule-to-LLVM callers default to the portable path.
LLVM legality is not used as a portability or cost claim: it only rejects
unsupported/scalarized shapes, while W8 profitability remains a documented
host measurement.
`LUISA_SIMD_DISABLE_PAIRED_LEAF_GATHER=1` is the same-binary oracle and
`paired_leaf_gathers` reports the number of emitted 64-bit gathers. Permanent
regressions cover W4/W8/W16 selection, a 13-element inactive tail, candidate/
oracle bit equality, LLVM intrinsic widths including an odd `uint3` tail, and
final x86 gather shape when the host proof succeeds.

The bounded grouping idea was independently derived after auditing ISPC
1.31.0 `GatherCoalesce` at revision `c6adb4f86f56` under BSD-3-Clause. No code
or coefficients are copied. Unlike ISPC's cross-instruction scan, this first
rule never crosses a Luisa resource-read instruction boundary.

### 4.8 Lane-consecutive typed-buffer and lane/value refinement

A direct, nonvolatile typed `BUFFER_READ` or `BUFFER_WRITE` may use an LLVM
masked contiguous vector access when Schedule lowering proves that its integer
element index has lane step one. The proof is use-site provenance; it does not
reclassify the SSA value. It currently recognizes `warp_lane_id`, the x
component of `thread_id`/`dispatch_id`, equal integer expressions, and step-one
plus equal add/subtract compositions. Select preserves a proof only when its
condition is equal across the cohort and both arms have the same step.

For dispatch/thread x, the static block-X dimension must be at least `W` and
divisible by `W`. Since packet starts are W-aligned inside one block, this
proves that the packet cannot cross an x row. If static geometry is missing or
the packet can cross a row, the index remains unannotated and uses the normal
gather/scatter path. Casts of a step-one value also fail closed until a
separate no-wrap/range proof exists. Byte-address, volatile, bindless,
structure, array, matrix, local, acceleration-structure, and ray-query
accesses are unchanged. A typed buffer's declared element must exactly match
the read result or written value.

The executing cohort is nonempty. Let `s = cttz(A)` be its first active lane
and `i_s` that lane's source element index. Lowering reconstructs the
conceptual lane-zero base as `b = i_s - s` in address-width modular integer
arithmetic. It must not read lane zero merely because lane zero is physically
present: a sparse cohort may leave that lane inactive with stale or invalid
state. The GEP is non-`inbounds`, and masked-off elements perform no memory
access. This makes partial tails and sparse masks observationally identical to
the original per-active-lane typed accesses. Scalar elements issue one
`llvm.masked.load` or `llvm.masked.store` at `b` with mask `A`.

One bounded lane/value axis rotation additionally accepts a top-level Luisa
vector with two to four non-Boolean 32-bit scalar components. Let `D` be the
semantic component count and `S = sizeof(vector) / 4` its physical storage
slots. Only `(D, S) = (2, 2), (3, 3), (3, 4), (4, 4)` with component offsets
`4c` are legal. The varying Schedule value is component-major, with one
`<W x T>` vector `V_c` per component. Generic `shufflevector` operations form
one storage-order vector `P` and memory mask `M` satisfying

```text
P[lane * S + component] = V_component[lane], component < D
M[lane * S + slot] = A[lane] && slot < D
```

One masked load/store operates on `<W * S x T>`, and loads deinterleave the
wide result back into `D` component vectors. For padded three-component
vectors, the fourth slot is always masked off: reads do not observe it and
stores do not overwrite it. Sparse cohorts and partial tails repeat each
logical lane bit only over that lane's semantic components. Thus the rewrite
does not make an inactive lane, padding byte, or adjacent buffer element
observable. Production IR contains no x86, AVX, AVX-512, NEON, or other target
intrinsic.

Inside a predicated direct-memory diamond, `A` in this construction is the arm
mask unless the index was already produced under the outer packet mask. In
that latter case the full-packet lane-step proof permits the outer safe seed;
the same algebra reconstructs `b`, while the memory access still uses the arm
mask. No such substitution is made for an arm-local index.

The scalar-element profitability policy remains W4/W8/W16; its measured W2
lowering remains disabled. The lane/value rotation is independently enabled at
W2/W4/W8/W16 after every width produced a positive paired path-trace gate. W1
already uses scalar memory and emits no transpose. The warp/cohort-uniform read
rule runs first, so a uniform vector index still performs one scalar aggregate
load and only then splats its result.

Permanent regressions cover aligned and row-crossing block geometry, disabled
lowering, W1 identity, W2/W4/W8/W16 selection, `uint4` LLVM IR and final
assembly without gather/scatter, a 13-element inactive tail, float2/float3/
float4 sparse cohorts, preserved float3 padding, and candidate/oracle exact
equality. Runtime reports all accepted accesses in `contiguous_buffer_reads`/
`contiguous_buffer_writes` and reports the vector subset separately in
`transposed_buffer_reads`/`transposed_buffer_writes`.
`LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER=1` is the same-binary A/B oracle for
both scalar and lane/value refinements.

### 4.9 Complete verifier-legal arithmetic lowering

For every `ArithmeticOp` and operand/result shape accepted by the XIR verifier,
Schedule-to-LLVM either emits the operation under this contract or reports a
specific capability failure before execution. The current arithmetic switch
has no accepted opcode that falls through to an unimplemented default.

The newly completed integer and scalar operations have these semantics:

- rotate-left/right use `llvm.fshl`/`llvm.fshr`; the shift is reduced modulo
  the integer bit width by the intrinsic semantics;
- `clz(0)` and `ctz(0)` return the integer bit width. Population count and bit
  reverse preserve the operand width;
- `step(edge, x)` is zero exactly when the ordered comparison `x < edge` is
  true and one otherwise, so an unordered NaN comparison returns one;
- `pow_int(base, exponent)` preserves the declared signedness and full width of
  the exponent. A signed negative exponent selects `1 / base`; its magnitude
  is formed with wrapping unsigned arithmetic, including the most-negative
  value. Exponent zero returns one.

`pow_int` is implemented by exponentiation by squaring. A uniform input invokes
one scalar helper. A varying input invokes one whole fixed-vector helper whose
loop terminates when `llvm.vector.reduce.or(exponent != 0)` is false; the loop
contains vector multiply, logical shift, and select, but no physical-lane
induction variable, `extractelement`/scalar-call/`insertelement` sequence, or
scalar libm dependency. Before the varying helper executes, inactive bases are
selected to one and inactive exponents to zero. Rotates and integer bit
operations similarly select benign inactive operands, so parked physical-lane
state is never fed to those intrinsics.

Source-vector `reduce_sum`, `reduce_product`, `reduce_min`, and `reduce_max`
fold the source aggregate's component axis. They never reduce across SIMD
thread lanes. Integer min/max preserves signedness; floating reduction order is
the deterministic component order, beginning with negative zero for sum and
one for product as in the scalar backend.

For dimensions two through four, vector outer product is
`result[column][row] = lhs[row] * rhs[column]`; matrix outer product is
`lhs * transpose(rhs)`. Transpose, determinant, and inverse are expanded over
SoA leaves. Determinants use cached recursive minors, and inverse uses the
transposed cofactor matrix divided by the determinant. A singular inverse has
no finite-result guarantee: division by a zero determinant and subsequent
non-finite propagation follow the configured LLVM floating arithmetic rather
than inventing a domain extension or invoking a host matrix routine.

Runtime semantic coverage uses independent host oracles at W1/W2/W4/W8/W16
and 35 threads, including zero count operands, signed and unsigned reductions,
the most-negative signed exponent, uniform NaN `step`, all matrix dimensions,
and a three-lane W16 tail. A direct XIR fixture additionally checks raw
fixed-vector rotate/power IR, absence of target-specific intrinsic namespaces
and lane extract/insert loops, absence of `powf` from optimized assembly and
object bytes, bit-exact active results, and untouched inactive-tail sentinels.

## 5. Vector-math providers

Provider selection is ordered:

```text
IR-native -> verified platform vector ABI -> unavailable
```

Warp-uniform operations use the scalar C/LLVM math operation exactly once.
A varying W1 specialization may use either one scalar operation or the shared
one-lane IR body. Varying `W = 2, 4, 8, 16` operations must use a native
provider. They do not fall back to `W` scalar calls.

### 5.1 IR-native baseline

The portable baseline is implemented as LLVM fixed-vector algorithms. Range
reduction, polynomial/rational approximation, bit classification, table
gathers, and exceptional-value repair are expressed without target-specific
intrinsics. The same algorithm is instantiated for W2/W4/W8/W16, then optimized
and legalized by the host target machine.

There are two semantic tiers:

| Shader option | Tier | Contract |
| --- | --- | --- |
| `enable_fast_math = false` | precise | documented ULP bound and IEEE special cases |
| `enable_fast_math = true` | fast | documented relaxed bound; no undefined domain extension |

The precise algorithms use SLEEF as the primary implementation and accuracy
reference. The fast tier is a distinct set of lower-order range reductions
and approximations; it does not acquire its semantics from whole-module LLVM
fast-math flags. Adapted source retains its upstream license and provenance.
The formula-level record is in
[`../common/LLVM_NATIVE_MATH_PROVENANCE.md`](../common/LLVM_NATIVE_MATH_PROVENANCE.md).

The current f32 implementation checkpoint is:

| Source operation | IR-native body | Regression bound/provider |
| --- | --- | --- |
| `sin`, `cos` | SLEEF-derived range reduction and polynomial | at most 4 ULP in the fixed boundary and deterministic bit-pattern corpus |
| `tan` | SLEEF-derived range reduction, polynomial, and quadrant reciprocal | at most 4 ULP in the fixed boundary and deterministic bit-pattern corpus |
| `asin`, `acos`, `atan` | SLEEF-derived domain transform and polynomial | at most 4 ULP in the fixed boundary, deterministic bit-pattern, and in-domain corpora |
| `exp` | SLEEF-derived range reduction and polynomial | at most 4 ULP in the fixed corpus |
| `log` | SLEEF-derived exponent reduction and polynomial | at most 5 ULP in the fixed corpus |
| `exp2` | direct nearest-integer base-2 reduction, split `ln(2)`, and native exponential polynomial | at most 1 ULP in the expanded independent corpus |
| `exp10` | direct `log2(10)` reduction, split `log10(2)`/`ln(10)`, and native exponential polynomial | at most 1 ULP in the expanded independent corpus |
| `log2`, `log10` | direct exponent/mantissa reduction and destination-base polynomial | at most 3 ULP in the expanded independent corpus |
| `atan2` | SLEEF-derived signed quadrant reduction and degree-17 odd polynomial | at most 4 ULP by contract; at most 2 ULP in the expanded paired corpus |
| `pow` | SLEEF-derived compensated `logkf`/multiply/`expkf` magnitude with explicit integer/domain repair | at most 4 ULP in the expanded paired corpus |
| `sinh`, `cosh`, `tanh` | locally derived near-zero series plus a shared overflow-safe `exp(x) / 2` primitive and reciprocal reconstruction | at most 8 ULP in the expanded corpus |
| `asinh` | locally derived odd series near zero, stable square-root/log identity, and large-magnitude logarithmic form | at most 8 ULP in the expanded corpus |
| `acosh` | locally derived series in `x - 1`, stable square-root/log identity, and large-magnitude logarithmic form | at most 8 ULP in the expanded corpus |
| `atanh` | locally derived odd series near zero and `log((1 + abs(x)) / (1 - abs(x))) / 2` elsewhere | at most 8 ULP in the expanded corpus |

SIMD Schedule lowering instantiates these bodies at W1/W2/W4/W8/W16. The
fallback backend uses the same provider for `sin`, `cos`, `tan`, `asin`,
`acos`, `atan`, `exp`, `exp2`, `exp10`, `log`, `log2`, and `log10` on DSL
float2/float3/float4 values, uses its binary providers for `atan2` and
`pow`, and uses the same six hyperbolic providers, so
component-vector math does not become two, three, or four scalar libm calls.
Its precise target options prohibit
aggressive FP contraction; helper functions are excluded from the later
whole-module fast-flag rewrite.

For a finite fast-tier result `y` and the reference `R`, the accepted error is
`abs(y - R) <= A + B * abs(R)`. The current contract is:

| Operation | Fast algorithm | `A` | `B` |
| --- | --- | ---: | ---: |
| `sin`, `cos` | three-term Cody-Waite for `abs(x) < 128`, precise SLEEF-derived reduction outside it, degree-9 odd polynomial | `2e-5` | `2e-5` |
| `tan` | the same common reduction, degree-11 odd polynomial, quadrant reciprocal | `2e-4` | `4e-4` |
| `asin`, `acos` | half-domain square-root transform, `asin` series through degree 7 | `2e-4` | `0` |
| `atan` | `pi/8` and `pi/4` identities, series through degree 9 | `1e-5` | `1e-5` |
| `exp` | nearest-integer base-2 reduction, degree-4 exponential polynomial | `2e-7` | `1e-4` |
| `exp2` | direct nearest-integer base-2 reduction and degree-4 exponential polynomial | `2e-7` | `1e-4` |
| `exp10` | direct `log2(10)` range reduction and degree-4 exponential polynomial | `2e-7` | `1e-4` |
| `log` | exponent extraction and an `atanh` series through degree 5 | `3e-6` | `1e-6` |
| `log2` | exponent/mantissa reduction combined directly in base two | `5e-6` | `2e-6` |
| `log10` | exponent/mantissa reduction combined directly in base ten | `2e-6` | `1e-6` |
| `atan2` | one min/max-magnitude ratio division and a locally derived degree-11 odd minimax polynomial | `3e-6` | `1e-6` |
| `pow` | fast `log(abs(base))`, exponent multiply, and fast `exp`, plus explicit domain/sign repair | `2e-7` | `5e-4` |
| `sinh`, `cosh` | lower-degree near-zero series plus overflow-safe fast `exp(x) / 2` reconstruction | `2e-7` | `2e-4` |
| `tanh` | lower-degree near-zero `sinh/cosh` ratio plus exponential reconstruction and saturation | `2e-5` | `2e-5` |
| `asinh` | degree-9 local odd series and one fast logarithm | `5e-6` | `2e-6` |
| `acosh` | degree-6 local `x - 1` series and one fast logarithm | `5e-6` | `2e-6` |
| `atanh` | degree-9 local odd series and one fast logarithm | `5e-6` | `2e-6` |

`exp2`, `exp10`, `log2`, and `log10` have independent provider symbols,
range reductions, standard-function references, and numerical bounds. They
are not audited as scaled `exp`/`log` compositions. For all three fast
exponential functions, the reference maps positive subnormal outputs to
`+0` as required by the fast-tier contract.

Fast special-value and domain behavior is defined, not inherited from LLVM
undefined fast-math assumptions:

- all NaN results use canonical quiet NaN `0x7fc00000`;
- `sin`, `cos`, and `tan` map either infinity to NaN; `sin` and `tan`
  preserve signed zero and every signed subnormal input, while `cos` maps
  those inputs to `+1`;
- `asin` and `acos` return NaN for `abs(x) > 1`; `asin(+-1) = +-pi/2`,
  `acos(+1) = +0`, and `acos(-1) = pi`. `asin` preserves signed zero and
  signed subnormal inputs; `acos` maps them to `pi/2`;
- `atan(+-infinity) = +-pi/2` and preserves signed zero and signed
  subnormal inputs;
- `atan2` returns canonical NaN when either argument is NaN; implements the
  signed-zero axes (`atan2(+-0, +x) = +-0`, `atan2(+-0, -x) = +-pi`, and
  `atan2(y, +-0) = copysign(pi/2, y)` for finite nonzero `y`); and handles
  finite/infinite and infinite/infinite pairs with the corresponding signed
  multiples of `pi/4`. Finite subnormal operands are evaluated without a
  deliberate flush-to-zero;
- `exp`, `exp2`, and `exp10` map either signed zero and every signed
  subnormal input to `1`, `-infinity` to `+0`, and `+infinity` to
  `+infinity`. Every positive subnormal result is flushed to `+0`;
- `log`, `log2`, and `log10` map either signed zero and every positive
  subnormal input to `-infinity`; negative finite values, negative
  subnormals, and
  `-infinity` map to NaN; positive infinity maps to positive infinity and
  an input of one maps to `+0`.
- `pow` returns canonical NaN for a negative finite nonzero base and a finite
  non-integer exponent, or for a NaN operand except the required
  `pow(x, +-0) = 1` and `pow(+1, y) = 1` identities. A negative result occurs
  exactly for a negative base and odd integral exponent. Zero and infinity
  bases, infinite exponents, and `pow(-1, +-infinity) = 1` follow the C/IEEE
  magnitude rules. Fast magnitude evaluation treats subnormal bases as signed
  zero and flushes subnormal results to signed zero, except that the exact
  `pow(x, 1) = x` path preserves every non-NaN input bit; a NaN base is still
  repaired to canonical quiet NaN. A negative subnormal base with a finite
  non-integer exponent remains a domain error rather than a real-domain
  extension.
- `sinh`, `tanh`, `asinh`, and `atanh` preserve signed zero and signed
  subnormal inputs. `cosh` maps them to `+1`. `sinh(+-infinity)` and
  `asinh(+-infinity)` return the correspondingly signed infinity;
  `cosh(+-infinity) = +infinity` and `tanh(+-infinity) = +-1`.
- `acosh(x)` returns canonical NaN for `x < 1`, `acosh(1) = +0`, and
  `acosh(+infinity) = +infinity`. `atanh(x)` returns canonical NaN for
  `abs(x) > 1`, including either infinity, and returns signed infinity at
  `x = +-1`. All NaN inputs and all hyperbolic domain errors return canonical
  quiet NaN; the fast tier does not extend either inverse-function domain.

Only FP contraction is permitted inside the fast bodies. They do not set
`nnan`, `ninf`, `nsz`, reassociation, approximate-function, or approximate-
reciprocal flags. This is what keeps the special-value contract meaningful.
Warp-uniform scalar expressions remain scalar and execute one LLVM/scalar
math operation; they are never splatted into the vector fast provider.

The regression instantiates precise and fast bodies at W2/W3/W4/W8/W16.
Every operation checks fixed boundaries and special values plus 8,192
deterministic raw float bit patterns, 8,192 domain-focused values, and 4,096
reduction/transition-focused values per width. The four independent
`exp2`/`exp10`/`log2`/`log10` bodies, binary `atan2`/`pow`, and all six
hyperbolic bodies raise those three
deterministic corpora to 65,536, 65,536, and 16,384 values respectively.
The radix operations include integer and half-integer reduction points,
exponent transitions, and both mantissa partition boundaries; `atan2` uses
paired raw patterns plus ratio, quadrant, axis, infinity, and magnitude
partitions. `pow` additionally covers near-one bases with large exponents,
negative-base integer parity, every special-value pair, and overflow/underflow
transitions. Hyperbolic focused samples include zero-series boundaries,
`x = 1` inverse-domain transitions, logarithmic large-magnitude transitions,
and the finite `sinh`/`cosh` overflow edge. Schedule tests cover
W1/W2/W4/W8/W16, scalar-uniform operations, and inactive tails.
Optimized assembly is rejected if it contains a varying scalar libm symbol.

#### 5.1.1 Fast-math XIR algebraic canonicalization

The fast tier now runs a separate conservative XIR pass before Schedule IR or
fallback LLVM lowering. It recognizes only f32 scalars and f32 vectors whose
required constant operand is bit-uniform across every component:

| Source | Replacement |
| --- | --- |
| `pow(x, +-0)` | `1` |
| `pow(+1, y)` | `1` |
| `pow(+2, y)` | `exp2(y)` |
| `pow(+10, y)` | `exp10(y)` |

These rules cover their complete input domain, including NaN, infinity,
signed zero, and subnormal operands, under the fast provider contract above.
They do not infer that an arbitrary base is positive and therefore cannot turn
a negative-base or signed-zero case into a real-domain extension. Instructions
carrying metadata are retained rather than moving or discarding an annotation.
The pass preserves the XIR type and uniformity class: a scalar uniform radix
power remains one scalar operation, while a varying scalar or component vector
selects the corresponding fixed-vector provider at its existing use site.

Even `pow(x, +1) -> x` is intentionally retained: a direct value bypasses the
provider's canonical-NaN repair when `x` is NaN. No generic
`pow(a, b) -> exp2(b * log2(a))`, `exp -> exp2`, `exp10 -> exp2`, or
inverse-composition folding is enabled. Those transformations change
special-value selection, intermediate overflow/underflow, or the audited
error envelope unless additional range facts are proved. The dedicated
`exp`, `exp10`, and logarithm providers already have direct range reductions,
so a change in spelling alone is not evidence of lower cost.

Precise mode is an explicit no-op. Pass tests cover scalar and component-vector
shape, both signs of zero, the safe IEEE identities, NaN-sensitive `x^1`
rejection, mixed vector constants, negative bases, f64 rejection, metadata
retention, and pass-pipeline option propagation. An AST-to-SIMD integration
fixture additionally proves that two
radix powers are rewritten only in fast mode; existing provider corpora supply
the raw-bit and special-value execution checks for the selected `exp2` and
`exp10` bodies. Optimized assembly remains subject to the scalar-libm-symbol
ban.

`benchmark_llvm_native_math --canonicalization-only` compares the rewritten
operation with the already-native fast `pow` provider using nine interleaved
samples per pair. On the LLVM 22.1.8 audit host, three complete runs measured
`pow(+2, x) -> exp2(x)` at 1.827x--1.962x and
`pow(+10, x) -> exp10(x)` at 2.565x--2.731x across W2/W3/W4/W8/W16. Every
width cleared the 1.05x gate and no scalar libm symbol appeared. The benchmark
reports entry-body instruction counts separately; throughput is the acceptance
metric because native helper bodies may remain outlined.

### 5.2 LLVM/system vector libraries

LLVM exposes vector-library selection through `TargetLibraryInfo`,
`ReplaceWithVeclib`, and `TargetOptions::VecLib` (including libmvec, SVML,
SLEEF GNU ABI, Accelerate, ArmPL, and AMD libm where supported). This is an
optional provider only. It is enabled for a function/width after checking:

- operating system and object ABI;
- library loadability and the exact symbol;
- detected ISA features required by that symbol;
- function, element type, width, masking semantics, and accuracy tier.

An LLVM mapping table is not itself a capability check. Unsupported widths
are explicitly chunked into supported *vector* widths or use IR-native code;
they never fall through to scalar libm calls.

### 5.3 Research basis

- SLEEF implements manually vectorized C99 real math functions, provides
  accuracy variants, and is distributed under Boost Software License 1.0:
  <https://github.com/shibatch/sleef>
- ISPC's default and fast math libraries are vector implementations; its
  system-math mode explicitly performs one scalar call per active instance and
  is not suitable for this backend's varying path:
  <https://ispc.github.io/ispc.html>
- ISPC 1.31.0 control-flow and memory optimizers were independently audited at
  source revision `c6adb4f86f56`. Its BSD-3-Clause implementation uses
  all/none/mixed mask paths, costed predication versus `any(mask)` arm skips,
  bounded gather coalescing, and constant-prefix masked-memory narrowing. These
  are design references only; no ISPC source or coefficient is copied into
  production:
  <https://github.com/ispc/ispc>
- Google Highway provides portable SIMD and a smaller contrib math surface;
  it is useful as a portability reference but is not the complete baseline:
  <https://github.com/google/highway>
- LLVM math intrinsics define vector semantics but do not promise a
  SIMD-native implementation:
  <https://llvm.org/docs/LangRef.html#llvm-sin-intrinsic>

## 6. Device-library and acceleration ABI

Every device-library operation has a capability tuple

```text
C = (operation, element type, W, target, mask support, accuracy/flags)
```

Codegen may select an implementation only if `C` is satisfied. A scalar C++
callback is permitted for W1 or an explicitly documented sparse fallback; it
is not completion for a normal W2/W4/W8/W16 path.

Indirect dispatch uses the shared `IndirectDispatchLayout` source ABI. A
backend-owned buffer contains one `uint32_t` authored-count word followed by
capacity records of seven words: logical size xyz, kernel id, and authored
group count xyz. Capacity is positive, no greater than `UINT32_MAX`, and cannot
be supplied through external memory. The encoded shader argument remains the
public logical capacity, while the SIMD packet ABI receives a view of the full
header-plus-record allocation. A non-indirect backend buffer is not a valid
source even if its bytes happen to match this layout.

`INDIRECT_DISPATCH_SET_COUNT` clamps the selected active value to capacity and
publishes it from the first active lane of the dynamic cohort; identical
uniform writes therefore issue one store. Differing active values remain an
unordered device data race.
`INDIRECT_DISPATCH_SET_KERNEL` writes only active in-range indices. Logical
size and kernel id are written for a valid index; a block size with any zero
component publishes three zero group-count words, otherwise the authored group
count is component-wise ceiling division. Before division or pointer
formation, inactive indices, logical sizes, and block sizes are selected to
zero, zero, and one respectively. Stores use the exact active-and-in-range
mask. Conflicting active lanes or packets that author the same word remain an
unordered device data race.

The host consumer first applies `plan_indirect_dispatch(capacity, offset,
maximum_count)`. It executes
`min(planned_command_count, authored_count)` records beginning at `offset`;
the authored count is relative to that host-selected range, matching the
Vulkan preparation contract. A record with any zero authored group component
is a no-op. Otherwise the consumer recomputes physical blocks from the logical
size and the target shader's block size, and supplies the record's kernel id in
the packet launch configuration. Authored group counts therefore validate the
writer's block size but cannot under-dispatch a target with a different block
size. Direct batched dispatch likewise assigns zero-based batch indices to
`kernel_id()`. SIMD stream command ordering joins the authoring dispatch before
host consumption, so no implicit asynchronous read is permitted.

Direct texture reads and writes cross the JIT/runtime boundary once per
packet, not once per lane. The packet ABI carries the active lanes as low bits
of a `uint64_t`, coordinates as three `lane_count`-element SoA arrays, and a
pixel as four consecutive component vectors. The runtime may service a
same-texel broadcast once, batch a fully active contiguous row, or iterate only
set bits for a sparse cohort. It must inspect no inactive coordinate and issue
no inactive resource access; read scratch is initialized before the callback
so an inactive tail cannot expose poison.

Direct 2D/3D sampling uses a distinct packet callback for `sample`,
`sample_level`, `sample_grad`, and `sample_grad_level`. It receives the bound
base mip, resource dimension, lane count and active bits, one packed explicit
sampler code per lane, SoA `u/v/w`, an optional SoA level, and four SoA result
components. The runtime groups active lanes by sampler code and invokes the
fixed-width sampler once per group. A uniform sampled result narrows the mask
to the first active lane; a varying result preserves the complete cohort and
tail mask. Coordinate, sampler, derivative, and level operands are selected to
benign values before scratch stores, conversions, native logarithms, or the
callback. The callback pointer is appended to the direct texture descriptor in
its existing 64-bit-host tail padding: the descriptor remains 64 bytes and all
established read/write/size field offsets remain unchanged.

The physical texture remains Luisa's native row-major resource layout. The
packet ABI is the SIMD-facing layout boundary: it permits coherence-aware
batching without changing native handles, upload/download layout, external
memory, or read/write synchronization. Generic JIT code must not inspect a
`SIMDTexture` or `FallbackTexture` C++ object layout. A backend-local bindless
descriptor may explicitly publish a raw view only together with the exact
storage/mip contract that makes it valid; every other texture path remains
behind the packet callback. Direct wide loads/gathers require a separate
pre-access bounds proof and performance gate.

For a fully active fixed-width packet whose 2D coordinates are one bounded,
increasing row span, native `FLOAT4` and `INT4` storage may use an
alignment-safe packet copy followed by an AoS-to-SoA transpose on reads, and
the inverse transpose on writes. The bounds check must prove the entire span
before the copy. Sparse masks, inactive tails, row crossings, 3D resources,
and every converting storage format retain the generic active-lane path. The
diagnostic environment flag
`LUISA_SIMD_DISABLE_CONTIGUOUS_TEXTURE_PACKETS=1` disables this specialization
for same-binary performance and fallback-path tests; it does not change the
public texture layout or semantics.

Bindless arrays extend the same packet boundary with a runtime-owned dense
slot table. Each slot contains independent buffer, 2D-texture, and 3D-texture
descriptors; a texture descriptor stores the resolved `SIMDTexture` plus a
raw mip-zero pointer when the storage is `BYTE1`, followed by a packed 64-bit
sampler/base-extent field. The backend-local descriptor is 24 bytes; two
texture descriptors plus the buffer view form one 64-byte slot aligned to 16
bytes. Non-`BYTE1` formats publish a null raw pointer. Each extent is limited
to twenty bits, and a bindless update fails before publishing the descriptor
if an extent does not fit. JIT code normally issues exactly one callback for a
varying packet and passes divergent slot indices through a SoA scratch array.
Before any scratch store, table lookup, coordinate conversion, mip conversion,
or sampler decode, inactive lanes are selected to benign zero operands.
Callback result storage is zero-initialized. The runtime then groups active
lanes that resolve to the same texture, sampler, and (where applicable) mip
before accessing the native row-major resource.

A result classified warp- or cohort-uniform invokes the callback for only the
first active lane and remains scalar after the callback. It is forbidden to
broadcast a uniform bindless query and repeat the resource operation W times.
The current supported bindless texture surface is 2D/3D `read`, `read_level`,
`size`, `size_level`, `sample`, `sample_level`, `sample_grad`, and
`sample_grad_level`, with either the sampler stored in the slot or an explicit
filter/address pair. Gradient LOD is
`0.5 * log2(max(dot(ddx * extent, ddx * extent),
dot(ddy * extent, ddy * extent), 1))`; the level operand is a minimum-LOD
clamp. Varying LOD uses the shared target-independent
fixed-vector native `log2` provider in JIT IR, with no scalar libm lane loop.
LOD uniformity is independent of sampled-color uniformity: when slot/extent
and both derivatives are warp- or cohort-uniform, extent decode, both squared
norms, `log2`, and an optional uniform minimum clamp execute once in scalar
SSA even if coordinates and the result vary. The scalar LOD is splatted only
at the packet callback ABI. A uniform sampled result separately invokes the
callback for only the first active lane. Gradient and extent operands are
sanitized before loads, masked gathers, or arithmetic, including an inactive
tail.

Mip behavior is explicit. A read whose integer level is outside the allocated
mip range returns the initialized zero pixel. A size query is computed from
the base extent as `max(base >> level, 1)`; levels at least the integer bit
width return one without performing an invalid shift. Bindless sampling uses
mip zero as its base; direct sampling uses the mip bound into the texture view,
and every explicit or derived LOD is relative to that base. Sampling without
an explicit level uses the base mip. A finite explicit level below zero uses
zero; finite positive levels clamp to the last mip available from the base,
`NaN` and negative infinity use zero, and positive infinity uses that last
mip. `POINT` and `LINEAR_POINT` select the nearest mip while `LINEAR_LINEAR`
and anisotropic mode interpolate adjacent mips. Gradient LOD uses the bound
base-mip extents. A zero gradient selects relative mip zero. A NaN in either
derivative produces derived LOD zero before an optional minimum-LOD clamp; a
non-NaN infinite derivative selects the last available mip. Non-finite sample
coordinates return the initialized zero pixel. Linear `ZERO` addressing keeps
the two mathematical taps distinct, so coordinates within half a texel of the
edge blend the in-range texel with the zero border; `REPEAT` likewise preserves
tap order across its wrap seam.

The common varying 2D `BYTE1`, mip-zero, stored
`LINEAR_POINT`/`MIRROR`, uniform-slot path is lowered directly in
target-independent fixed-vector JIT IR. The descriptor is loaded once; mirror
range reduction, coordinate conversion, four tap addresses, interpolation,
and result masking remain vectors. Inactive or non-finite coordinates are
selected to benign values before float-to-integer conversion and before any
gather, and produce the initialized zero pixel. All other samplers, storage
formats, mip/gradient/explicit-sampler operations, divergent slots, and
uniform sampled results retain the packet callback and its existing semantics.

The baseline direct tap load is an alignment-one masked byte gather. At W4 and
W8 only, a measured specialization may instead issue alignment-one 32-bit
gathers and retain the byte at the addressed lowest memory location. Before
that wider access, JIT IR proves for all four taps and every participating lane
that `offset <= width * height - 4`; inactive lanes do not constrain the
proof. A texture smaller than four bytes, any packet touching the final three
bytes, or a failed proof takes the narrow gather path before memory is
accessed. Big-endian targets shift the loaded word before masking. This changes
neither public row-major layout nor external-memory requirements. W1/W2/W16
remain narrow because the measured wide form was neutral or slower there.
`LUISA_SIMD_DISABLE_IR_BYTE1_TEXTURE_SAMPLING=1` restores the complete callback
oracle, while `LUISA_SIMD_DISABLE_WIDE_BYTE1_GATHERS=1` keeps direct JIT
sampling but forces narrow gathers.

Embree traversal uses the packet API matching the specialization width:

| Width | Trace | Occlusion | Validity |
| ---: | --- | --- | --- |
| 1 | `rtcIntersect1` | `rtcOccluded1` | scalar active lane |
| 2 | `rtcIntersect4` | `rtcOccluded4` | lanes 0--1 from the cohort; lanes 2--3 invalid |
| 4 | `rtcIntersect4` | `rtcOccluded4` | 4-lane valid mask |
| 8 | `rtcIntersect8` | `rtcOccluded8` | 8-lane valid mask |
| 16 | `rtcIntersect16` | `rtcOccluded16` | 16-lane valid mask |

Ray/hit state is stored in packet-compatible structure-of-arrays form. The
current cohort mask and dispatch tail jointly form Embree's valid mask.
Inactive rays are initialized to benign values even though the validity mask
excludes them. LLVM performs that sanitization before the callback and
initializes every public result field. The scratch uses Embree's public
`RTCRayN`/`RTCHitN` component order; compile-time `sizeof`, `alignof`, and
`offsetof` checks prove the configured Embree headers match the shared field
indices. Direct trace does not expose Embree's application-defined `ray.id`,
so LLVM sign-extends the callback-valid mask into that packet field. For native
W4/W8/W16, the runtime aliases the field as Embree's `const int *valid`, passes
the same scratch directly to Embree, and receives results in place. No packed
mask expansion, intermediate ray packet, or hit copy is permitted. W1 checks
the embedded scalar validity before using the same in-place layout with the
scalar API. W2 alone copies its two-lane fields into a zero-padded W4 packet,
copies validity into the padded Embree mask, and copies the public fields back;
it may not read beyond the two-lane source scratch.

Embree initializes direct-query arguments for incoherent rays. Production keeps
that default for W1/W2/W4/W8, every sparse or partial W16 packet, and every
stateful ray query. A direct W16 closest-hit or occlusion call whose sixteen
valid entries are all active instead sets Embree's coherent-ray flag. This is a
runtime packet-shape specialization, not a promise that arbitrary active rays
have equal origins or directions; its correctness contract remains the same
public closest/any result contract. Embree may choose a different valid
traversal order, so numerically tied or near-tied hits are allowed the same
floating-point variation already admitted across its traversal algorithms.
`LUISA_SIMD_DISABLE_COHERENT_W16_DIRECT_TRACE=1` restores the complete
incoherent direct-trace oracle. The default and oracle acceleration tests both
cover two full W16 packets and a three-lane W16 tail; the tail must never receive
the coherent flag.

This reuse is restricted to direct trace/occlusion. Ray queries retain physical
lane IDs in `ray.id`: candidate filters use them to recover lane-private query
state. A permanent callback probe covers sparse tails at W1/W2/W4/W8/W16 and a
non-contiguous W8 `0x55` cohort, including inactive operand sanitization and
returned closest/any fields.

Direct closest-hit traversal sets `bary.y = -1` for round curves after Embree
returns. Each normal accel build recomputes whether its current instance table
contains any curve, including after primitive replacement or instance-count
shrink. A curve-free accel skips this per-active-lane geometry-kind scan; an
accel containing at least one static or motion-instance curve must retain the
scan for every direct closest-hit packet. This summary is a performance gate
only and does not change the stable instance table, valid mask, or hit ABI.
Permanent coverage replaces one instance `mesh -> curve -> mesh` and checks
classification after every rebuild at W1/W2/W4/W8/W16.

Ray queries use the same width mapping, but retain state across candidate
handlers. The AST `RayQueryLoop` is first lowered to ordinary XIR
loop/if control containing `PROCEED`, candidate-kind reads, and object writes;
the existing cohort scheduler therefore executes divergent handlers without a
second callback-side PC machine. Query construction is always varying even
when its accel, ray, time, and visibility inputs are uniform: each physical
lane receives a distinct 16-byte-aligned fixed-size state record, while the
uniform input expressions themselves are still evaluated only once and splat
only at the state initialization stores. A copied query value is an internal
pointer to that lane's record.

Distinct simultaneously live query objects also receive distinct records.
Sequential construction sites may share the same per-lane scratch only after
a fail-closed Schedule-IR analysis proves that each construction result is
stored into exactly one unaliased query local. Backward liveness over every
branch, switch, convergence, loop-back, barrier-resume, and return edge builds
an interference graph; greedy coloring assigns scratch slots. A copied query,
an ambiguous local root, or an unrecognized use disables coloring for that
shader. Different scheduler cohorts may safely share a colored slot because
their active masks contain disjoint physical lanes; each construction writes
only its active lane records. Permanent regressions cover two sequential
queries sharing one slot, mutually exclusive divergent queries sharing one
slot, two overlapping queries retaining two slots, and an inactive W8 tail.
The disable control above restores one slot per construction without changing
query semantics.

`PROCEED` gathers the active state pointers into one `<W x ptr>` scratch,
selects null for inactive lanes, and issues exactly one indirect host callback
for the current cohort. The callback groups states by accel and query-all/
query-any mode, then uses W1 scalar traversal, W2 padded W4 traversal, or the
matching W4/W8/W16 packet entry. No active-lane extract/call/insert traversal
loop is permitted. Query-all and query-any support triangle and round-curve
surface candidates plus procedural candidates, reject-by-return, surface and
procedural commit, explicit terminate, world-ray reads, committed-hit reads,
static time zero, and motion time. Committing immediately updates the public
world-ray `t_max`; query-any terminates after the first commit. An opaque
surface instance auto-commits without entering the surface handler. Miss state
begins with invalid instance/primitive IDs, `HitType::Miss`, zero
barycentrics, and zero committed distance.

Surface candidate enumeration uses a bounded speculative batch. During one
Embree traversal, the argument filter rejects physical candidates while
retaining the nearest 32 `(t, instance, primitive)` keys after each lane's
cursor directly in that lane's persistent query state. Arrival order is not
assumed: the runtime keeps the closest 32 with a max heap only after overflow,
then publishes the batch in ascending lexicographic order. A later `PROCEED`
still crosses the normal packet callback boundary, but consumes the next cached
candidate without traversing Embree. Commit updates `t_max` immediately and
invalidates cached candidates beyond it; terminate and opaque auto-commit stop
the lane without consuming the rest of the batch.

For a round curve, Embree may report both front and back surfaces after a filter
rejects the first hit. The batch keeps only the nearest candidate for each
curve `(instance, primitive)` pair, and a continuation scan suppresses that
pair after it has been published once. This realizes Luisa's one-candidate-per-
curve-primitive contract without adding the O(N) duplicate check to triangle
insertion. Direct and query hits preserve Embree `u` as the curve parameter and
set `bary.y = -1` as the public curve discriminator.

Procedural resources use Embree user geometry whose public AABB buffer is a
conservative bound, not a physical hit. The intersect and occluded callbacks
always reject the Embree hit. During an active query scan, a thread-local scope
identifies the exact runtime query context and records `(instance, primitive)`;
outside that scope, including direct closest/any traversal, user geometry
remains a miss. The generated handler executes only after packet traversal
returns to the ordinary cohort scheduler. Embree occlusion traversal may
conservatively invoke a user callback even when the ray does not intersect the
exact AABB; such a callback is a rejectable procedural candidate and cannot by
itself create a committed hit.

Procedural candidates have a separate 32-entry speculative batch ordered by
`(instance, primitive)`. Duplicate callback invocations sort adjacently and the
cursor publishes each key at most once. Rejecting a candidate consumes the
cache without another BVH traversal. A surface or procedural commit changes
`t_max`, so every unexposed conservative procedural candidate is discarded;
cached exact surface hits remain and are interval-filtered, and a continuation
scan is requested only when discarded or overflow candidates may remain. More
than 32 procedural keys use another scan strictly after the cursor.

The fixed state ABI is 1216 bytes per lane: it contains 32 surface hits, 32
procedural keys, independent cursors, and explicit metadata for both batches.
At query construction, `candidate_batch_initialized` and
`procedural_batch_initialized` are the only readable batch gates. W1/W4/W8/
W16 therefore leave the corresponding count, index, and continuation fields
uninitialized until the first scan, which clears all six fields before either
initialized bit is published. `advance_ray_query_candidate` must test both
gates before reading any of those fields. W2 retains eager initialization: its
padded-W4 renderer measurements did not establish a stable benefit from the
lazy path. The eager oracle is selected with
`LUISA_SIMD_DISABLE_RAY_QUERY_LAZY_BATCH_INIT=1`. Exact LLVM tests count
masked-scatter callsites rather than intrinsic declarations: lazy/unpacked
construction requires 31 and eager/unpacked construction requires 37.
At W4/W8/W16, five adjacent equal-bit-pattern field pairs are initialized by
five 64-bit masked scatters instead of ten 32-bit scatters: candidate kind/
commit flag, terminated/procedural-cursor-valid, committed instance/primitive,
committed barycentrics, and committed kind/distance. The first two addresses
carry their real four-byte alignment; committed-hit addresses are eight-byte
aligned. Only all-zero or all-one pairs are packed, so the representation is
independent of host byte order and preserves positive-zero floats. W1/W2 keep
the unpacked stores because their renderer A/B did not establish a benefit.
`LUISA_SIMD_DISABLE_RAY_QUERY_PACKED_INIT=1` selects the unpacked oracle.
Inactive lanes remain excluded by the same construction mask. The final exact
counts are 31 for lazy/unpacked W1, 37 for eager/unpacked W2, 26 for packed
W4/W8/W16, 32 for the W8 eager/packed oracle, and 31 for the W8 lazy/unpacked
oracle.
The permanent surface 35-candidate and procedural 40-candidate regressions
cross the respective boundaries at W1/W2/W4/W8/W16 and require exactly-once
ascending handler delivery before the final commit. This removes repeated
traversal for the common bounded case without moving handler execution or its
control flow into the runtime.

At W4/W8/W16, an eligible query local may have one JIT-owned packed status
sidecar. Bits `[0, 16)`, `[16, 32)`, and `[32, 48)` respectively represent
terminated, surface-candidate, and procedural-candidate physical lanes;
`[48, 64)` is the initialization-valid mask. These fields are independent.
In particular, explicit terminate must not clear a still-observable candidate
kind. W1/W2 and any query whose ownership, construction store, or aliasing is
not proven continue to gather the authoritative fields from the 1216-byte
state. Disabling query scratch coloring also disables this refinement.

Construction may prepare state and callback data before the query pointer is
stored, but it must not publish the sidecar valid bit until after that masked
local store. A status update clears and replaces exactly the current active
lanes in all four fields, leaving other cohorts in a shared color untouched.
Every cached read must trap if any active lane is invalid, then mask the result
by the current active cohort so stale inactive bits remain unobservable. A
terminate update merges the active terminated bits. Candidate commits may use
the cached kind only for validation/masking; payload and interval data remain
authoritative in the public state.

The same ownership, alias, construction-store, and scratch-color proof may
also allocate one JIT-owned fixed-vector state-handle cache per status color.
The authoritative pointer is still stored in the lane-private query local.
Only after that masked store completes may lowering merge the written pointer
into the cache under the current active mask. Every use checks that all active
cached handles are non-null before passing a null-sanitized packet to the host
callback. Lanes outside the current cohort may retain another definition's
handle, but cannot be read or passed as active. W1/W2, an unproven query, a
disabled status cache, or disabled scratch coloring must retain the original
query-local gather path. The cache does not change the public state, accel
descriptor, or Embree ABI. `LUISA_SIMD_DISABLE_RAY_QUERY_STATE_HANDLE_CACHE=1`
is its same-binary semantic/performance oracle.

The status-aware host entry and the construction-selected plain callback form
one internal ABI pair. The entry must invoke that provider, preserving generic
versus triangle-only and narrow versus wide selection; the provider must fail
closed if any active non-null state stores a different plain callback. The
entry may inspect only active non-null state pointers afterward. It returns one
scalar packed mask and must not replace W4/W8/W16 Embree packet traversal with
per-lane scalar calls. JIT lowering always verifies that every active state has
the same status callback. When the status ownership/color proof holds, it may
rely on the paired provider for the plain-callback check and omit the redundant
masked gather; all other paths verify both callbacks in JIT. The stable
instance-table descriptor owns the status-entry pointers; the six-pointer
accel argument and the public query-state layout do not change.
`LUISA_SIMD_DISABLE_RAY_QUERY_STATUS_CACHE=1` is the same-binary semantic/
performance oracle, while
`LUISA_SIMD_DISABLE_RAY_QUERY_STATUS_CALLBACK_PAIRING=1` restores only the
redundant JIT check.

A W16 device may install a specialized paired status entry only when the
acceleration structure's latest complete build summary contains a procedural
instance and `LUISA_SIMD_DISABLE_PROCEDURAL_WIDE_STATUS_PACK` is unset. The
summary and entry selection must be refreshed after every build; W1/W2/W4/W8
and acceleration structures without procedural instances must retain the
generic entry. The specialization must invoke the same plain provider and
preserve its fail-closed callback-agreement validation. After that call, an
exact physical mask of `0xffff` may be packed by a sequential sixteen-entry
pass. Any other mask must use an inactive-safe sparse scan; no inactive or null
state pointer may be dereferenced. Candidate-kind, termination, reserved bits,
and high input-mask bits have exactly the generic packer's interpretation.
This refinement changes neither the query-state ABI nor the W16 Embree packet
width and must remain independently disableable from JIT status caching and
callback pairing.

The production W16 procedural entry fuses status publication into the two
passes that already finalize a query step. A lane satisfied from an existing
candidate batch publishes immediately after advance; a lane requiring Embree
traversal publishes while the newly scanned candidate batch is sorted,
installed, and advanced. No post-proceed state scan is permitted on this path.
The status still describes the exact final public state: pending lanes publish
only after traversal, terminated and candidate-kind fields remain independent,
and inactive/null lanes are never inspected. The entry validates every active
state against the same plain W16 provider, retains `rtcIntersect16`/
`rtcOccluded16`, and supports multiple accel/terminate-mode groups in one
cohort. `LUISA_SIMD_DISABLE_PROCEDURAL_WIDE_FUSED_STATUS=1` restores the
previous paired-call-plus-pack implementation as a same-binary semantic and
performance oracle. The permanent procedural regression executes W16 with
both entries, including an inactive tail, divergent commit/reject/terminate,
and a continuation beyond the 32-candidate batch.

The fused installer may bias its already-ascending batch path only while the
builder invariant `heapified => !ascending` holds. The non-ascending edge must
still distinguish heap restoration, a strictly descending reversal, and a
general sort, and must produce the same lexicographic publication order as the
plain provider. This is a control-layout refinement, not permission to skip
ordering validation or extend the query domain.

Each normal accel build recomputes whether the complete current instance table
contains a curve or procedural primitive. Motion instances are classified by
their child. When both summaries are false, the host view may select the
triangle-only ray-query provider; otherwise it must select the generic
surface/curve/procedural provider. Provider selection happens whenever an
accel view is encoded, so replacing an instance and rebuilding may change the
provider without recompiling the shader. The diagnostic control
`LUISA_SIMD_DISABLE_TRIANGLE_ONLY_RAY_QUERY=1` is sampled once at accel
construction and forces the generic provider for that accel.

The triangle-only provider is a semantic specialization, not a smaller state
ABI. JIT construction and callbacks still exchange the same 1216-byte state,
and every public field retains its existing meaning. The specialized runtime
must not read or write procedural batch/cursor fields, load geometry kind, or
run curve front/back deduplication. Its private Embree context contains only
the RTC context, lane count, active state pointers, and surface batch-build
metadata; only active entries need initialization. Candidate order,
32-candidate overflow continuation, interval/cursor tests, commit and query-any
termination, instance opacity, visibility, ray time, signed-zero `t_min`
handling, and inactive-tail safety are identical to the generic provider.
W1 uses one scalar Embree traversal, W2 one padded W4 traversal, and W4/W8/W16
one matching native packet traversal per scan. Curves or procedural instances
must never enter this path, including through a motion child.

At W8/W16, a full runtime cohort mask must use a dense proceed loop in either
the generic or triangle-only adaptive provider. A sparse mask may iterate its
set bits and may initialize/install only the corresponding lane records, but
it must retain the physical Embree packet width and pass the exact sparse valid
mask to one packet traversal. This is a runtime refinement of a varying cohort,
not a promotion in the static `warp_uniform -> cohort_uniform -> varying`
lattice. If a varying control-flow region happens to reconverge to a full
cohort, the next proceed call must take the dense path automatically. W1/W2/W4
use the dense specialization and do not carry adaptive control flow. The JIT-
side choice between dense and adaptive callback remains fixed by specialization
width; the accel-host-view choice between generic and triangle-only callbacks
is fixed for each encoded accel view. A distinct-provider LLVM boundary
regression covers W4 and W8. No inactive state pointer or operand may be
inspected before its mask check, and overflow/cursor ordering is identical on
both paths.

For a W8/W16 packet, the Embree surface filter may first convert the
fixed-width `valid` array into an integer mask and iterate only its set bits.
This includes sparse callback masks produced by Embree from an initially full
cohort. It is an implementation refinement of the same packet callback: it
must clear every visited Embree-valid entry, preserve candidate ordering,
cursor tests, and overflow behavior, and must not inspect any inactive
query-state pointer. The generic filter additionally preserves curve-primitive
deduplication; the triangle-only filter is allowed to omit it only under the
accel summary above. Every W1/W2/W4 packet uses a dense filter. Each filter
context places the configured Embree context at offset zero, as required by
Embree callback recovery; the triangle-only context is an independent private
type rather than an unrelated layout-compatible cast of the generic context.

The currently accepted acceleration surface is static and vertex-motion
triangle-mesh build; static and control-point-motion round-curve build for
piecewise-linear, cubic B-spline, Catmull--Rom, and Bezier bases; static and
AABB-motion procedural build; top-level static and mesh/curve/procedural
motion-instance build; affine transform; visibility mask;
`RAY_TRACING_TRACE_CLOSEST`; `RAY_TRACING_TRACE_ANY`; surface/procedural
`RAY_TRACING_QUERY_ALL`/`RAY_TRACING_QUERY_ANY`; and their motion-blur variants.
Direct closest/any deliberately rejects procedural geometry because no public
intersection handler exists on those operations.
Instance transform, user-id, visibility-mask, MATRIX-motion, and SRT-motion
queries are also accepted. A static trace/query uses time zero. A motion
operation passes one f32 time vector sanitized under the cohort mask before the
callback. A normal closest/any result classified warp- or cohort-uniform invokes
only the first active lane and stays scalar; mutable ray-query state is the
explicit varying exception described above. Closest-hit scratch starts with
invalid instance/primitive IDs and zero barycentrics/distance; occlusion scratch
starts false. Therefore an inactive lane cannot observe poison or mutate query
state even if complete initialized vectors cross the callback boundary.

Instance metadata uses a stable runtime-owned table descriptor whose data
pointer and count are republished after any build that can reallocate storage.
Transform, user-id, and visibility-mask reads and writes are supported.
Uniform instance operations issue scalar loads/stores. Varying operations
select IDs and values to benign zero under the inverse active mask before
bounds checks, stride multiplication, pointer formation, conversion, masked
gather, or masked scatter. Any active out-of-range ID traps. The stored 3x4
row-major affine is explicitly transposed and extended with `(0, 0, 0, 1)` on
read, and the inverse layout conversion is applied on write. Visibility writes
truncate to the public eight-bit mask contract. Every write marks the instance
dirty; a subsequent normal accel build commits dirty transforms and masks to
Embree before traversal. Conflicting active lanes that write the same instance
remain a device data race, as in other unordered device writes. JIT code must
not alias the private runtime object or C++ vector layout.
Every accel build also recommits each instance geometry before the TLAS scene,
even when its metadata is clean, because a rebuilt child mesh, curve, or
procedural scene can change bounds without changing the parent instance record.

`update_instance_buffer_only` updates the stable instance table but must not
mutate or commit the Embree TLAS. Host modifications and prior device metadata
writes remain marked dirty. Desired primitive bindings and owned motion frames
survive independently of the committed Embree geometry vector, so consuming a
buffer-only command cannot lose a newly appended or replaced primitive. Before
the next ordinary build, metadata queries observe the updated table while ray
traversal observes the last committed scene. The ordinary build must reconcile
geometry count and every dirty primitive/transform/mask even if its command has
no modifications, then commit geometry and scene and clear the dirty state.
Using a size- or primitive-changing table for traversal before that build has
the same intentionally stale-BVH semantics as the public buffer-only API; no
backend may silently perform the missing build on the user's behalf.
Geometry kind, committed count, and query-provider selection must continue to
describe that stale BVH. In particular, a buffer-only primitive replacement
must not reinterpret old curve/procedural hits as the desired new kind, and a
shrink must retain enough committed classification for an old instance ID.
Current in-range opacity is read from the newly published table; an ID removed
from that table uses its last-built opacity until the BVH shrink is committed.
Metadata-only buffer updates must not rescan geometry-kind summaries unless
the instance count or a primitive binding changes.

Motion-instance resources accept two through `RTC_MAX_TIME_STEP_COUNT`
keyframes, a finite strictly increasing time range, and MATRIX or quaternion
SRT mode. A motion resource must be built before it is inserted into a TLAS.
The TLAS owns a copy of its 64-byte public keyframes and exposes only a stable
frame pointer, count, and mode through the plain instance table. MATRIX frames
are composed with the instance's outer affine before being supplied to Embree.
SRT frames map pivot/quaternion/scale/shear/translation to
`RTCQuaternionDecomposition`; Embree normalizes each nonzero quaternion and
performs quaternion interpolation. An identity outer affine is represented by
one native Embree instance. A nonidentity outer affine must not approximate the
composition by interpolating endpoint matrices. It is represented by one
top-level user geometry whose private native SRT helper supplies the exact
Embree transform at the active ray time. The callback composes `outer * SRT`,
checks that a finite inverse is representable, inverse-transforms origin and
direction without normalization so public `t` is invariant, traverses the
child scene once at the native packet width, and inverse-transpose transforms
the committed geometric normal. Embree 4 uses its forwarding ABI; Embree 3
uses the documented recursive `rtcIntersect1/4/8/16` or
`rtcOccluded1/4/8/16` call with an explicit context instance-ID push/pop. W2 is
padded into one W4 call. Only W1 may call a scalar child traversal. A non-finite
ray time, singular composition, or composition whose finite inverse cannot be
represented is a miss; inactive lanes do not evaluate the inverse. Empty child
bounds remain empty. A time range beginning inside the camera shutter requires
`should_vanish_start`, and a range ending inside it requires
`should_vanish_end`, matching Embree's disappear-outside-range behavior instead
of inventing endpoint clamping.

Each motion resource build advances a host generation. A later ordinary TLAS
build must import a newer generation and mark the corresponding instance dirty
even when the Accel command carries no modifications. Without a newer host
generation, device-authored keyframe writes in the TLAS-owned copy remain
authoritative. The committed user-geometry callback payload has scene lifetime:
an `update_instance_buffer_only` transform change may update the public table,
but it must retain the old route and payload until the next ordinary build.
That build may replace USER with native INSTANCE, or the reverse, at the same
geometry ID before committing traversal state.

MATRIX/SRT keyframe reads and writes use the same scalar-or-masked policy as
static metadata. Both instance and keyframe IDs are sanitized before checks or
address formation; active IDs must be in range and the requested mode must
match the stored resource. A uniform instance with varying keyframes loads its
frame descriptor once and splats only the pointer/count metadata. Writes mark
the owning TLAS instance dirty, and a subsequent normal accel build validates
the frame values, republishes every Embree time step, and commits the scene.
Conflicting active lanes that write the same keyframe remain an unordered
device data race. No host callback or per-lane extract/call/insert loop is used
for keyframe metadata.

`RAY_TRACING_SET_INSTANCE_OPACITY` writes the stable instance table without a
host callback. The value type is exactly `bool`. A warp- or cohort-uniform
instance/value pair performs one scalar store; a varying pair selects both the
instance index and value to benign zero before bounds checking, address
formation, or a masked scatter. The ABI byte is the zero-extended boolean, so
only zero or one may be stored. Every active write sets the same dirty byte as
other instance metadata writes. An opaque surface query auto-commits and skips
the surface handler; a non-opaque query publishes the candidate to the handler.
Inactive lanes neither participate in bounds checks nor touch the opacity or
dirty bytes. Conflicting active writes to the same instance remain a device
data race.

All Embree scenes in one backend module share a single `RTCDevice`. If that
device reports the oneTBB tasking system, backend teardown must quiesce the
attached task scheduler after releasing the device and before `dlclose` can
unmap libtbb. Repeated device creation/destruction in one process is a required
lifecycle regression, not merely a leak check.

Deeper public instance-stack behavior beyond this one logical outer-SRT-child
composition is not part of this slice. It must fail at a specific capability
boundary until its independent semantic, ABI, and machine-boundary gates exist;
triangle, curve, procedural-query, opacity, and outer-SRT support do not imply
arbitrary nested instancing.

## 7. Executable audit matrix

Every newly supported device-library operation is accepted at three layers:

1. **semantic**: active lanes are compared with an independent scalar oracle,
   including domains, signed zero, infinities, NaNs, tails, and divergent
   masks;
2. **IR shape**: W2/W4/W8/W16 contain fixed-vector operations or one approved
   vector ABI call, with no lane-enumeration loop;
3. **machine boundary**: optimized object/assembly has no forbidden scalar
   math/device symbol and uses only ISA features reported by the target
   machine.

Uniform fixtures separately prove that kernel parameters and other uniform
seeds stay scalar and issue at most one call. Performance benchmarks measure
throughput but do not replace any correctness layer.

Every counterexample found by this audit is first retained as a regression
that fails the old lowering, then fixed. Provider availability never changes
source semantics: an unavailable native implementation produces a compilation
diagnostic rather than silent scalarization.
