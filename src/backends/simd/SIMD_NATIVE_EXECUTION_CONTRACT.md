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
the whole reachable Schedule CFG cannot split a cohort: it has no convergence
point, and every conditional or indexed selector is `warp_uniform` or
`cohort_uniform`. The LLVM function then contains direct scalar branches around
fixed-vector values. Its active mask is the immutable dispatch-tail mask;
cohort-uniform state remains scalar across blocks, and varying memory/effects
remain predicated by that mask. A cohort-uniform selector may choose a
different edge for different packet invocations, but never chooses different
edges for active lanes in one packet.

This proof is fail-closed. Any varying selector, convergence point, or
unsupported terminator retains the general worklist scheduler. Separately, the
general scheduler may discover at runtime that every active lane of one
varying branch chose the same successor and directly thread that edge; this
does not make subsequent control statically direct or discard scheduler state.
`LUISA_SIMD_DISABLE_COHERENT_DIRECT_CFG=1` forces the scheduled implementation
for differential diagnostics. Permanent tests cover all widths, partial tails,
cohort-uniform branches with packet-dependent outcomes, and the forced fallback.

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
`LUISA_SIMD_DISABLE_COHERENT_DIRECT_CFG=1` forces otherwise coherent functions
through the general cohort scheduler;
`LUISA_SIMD_DISABLE_AGGREGATE_PROMOTION=1` restores aggregate local storage
before the two SROA/`mem2reg` stages;
`LUISA_SIMD_DISABLE_UNIFORM_BUFFER_BROADCAST=1` controls the typed-buffer
refinement, and `LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER=1` controls proven
lane-consecutive typed-buffer accesses. `LUISA_SIMD_REPORT_OPTIMIZATIONS=1`
logs per-shader transform, scheduler-state, ray-query scratch, ray-query
status-color, and cached state-handle counters.
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
for the two state-layout refinements.
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

### 4.5 Bounded loop-unswitch refinement

The production SIMD compiler may replace a repeated internal conditional with
one conditional before two cloned loop versions. This is permitted only when
all of the following hold:

- the destructured natural loop is innermost, has one preheader, latch, exit
  edge, and exit block, and has a statically proven trip count greater than
  one;
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
edge. The old preheader dispatches to two canonical preheaders. Existing exit
PHIs receive the cloned incoming edge; direct live-outs receive one new exit
PHI. Candidate branch metadata moves to the outer dispatch, and structured CFG
causes atomic rejection before mutation. Rejecting unknown/zero-trip loops is
semantic, not merely a cost choice: it prevents consuming a selector on a lane
where the source branch would never execute. Rejecting writes and clocks
prevents the cohort-order change from becoming observable.

The outer selector still follows the ordinary scheduler rule. If its active
values happen to agree at runtime, the dynamically coherent fast path directly
enters one specialized loop; otherwise the packet splits once and each cohort
executes its selected version. Inactive tail lanes remain outside both masks.
The rewrite introduces no speculative arm evaluation and does not relax the
operand-sanitization requirement.

Correctness gates cover cloning, cyclic PHIs, exit-PHI and direct-live-out
repair, metadata, structured-module atomicity, unknown trip counts, `undef`,
clock/write rejection, all supported SIMD widths, and inactive tails.
`benchmark_simd_loop_unswitch` additionally audits optimized assembly, calls,
stack references, and repeated throughput.

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

### 4.7 Lane-consecutive typed-buffer refinement

A direct, nonvolatile typed `BUFFER_READ` or `BUFFER_WRITE` of a scalar Luisa
element may use one LLVM masked contiguous vector access when Schedule
lowering proves that its integer element index has lane step one. The proof is
use-site provenance; it does not reclassify the SSA value. It currently
recognizes `warp_lane_id`, the x component of `thread_id`/`dispatch_id`, equal
integer expressions, and step-one plus equal add/subtract compositions.
Select preserves a proof only when its condition is equal across the cohort
and both arms have the same step.

For dispatch/thread x, the static block-X dimension must be at least `W` and
divisible by `W`. Since packet starts are W-aligned inside one block, this
proves that the packet cannot cross an x row. If static geometry is missing or
the packet can cross a row, the index remains unannotated and uses the normal
gather/scatter path. Casts of a step-one value also fail closed until a
separate no-wrap/range proof exists. Byte-address, volatile, aggregate, and
bindless accesses are unchanged.

The executing cohort is nonempty. Let `s = cttz(A)` be its first active lane
and `i_s` that lane's source element index. Lowering reconstructs the
conceptual lane-zero base as `b = i_s - s` in address-width modular integer
arithmetic, then issues `llvm.masked.load` or `llvm.masked.store` at `b` with
mask `A`. It must not read lane zero merely because lane zero is physically
present: a sparse cohort may leave that lane inactive with stale or invalid
state. The GEP is non-`inbounds`, and masked-off elements perform no memory
access. This makes partial tails and sparse masks observationally identical to
the original per-active-lane typed accesses.

The current profitability policy enables the transformation only for
W4/W8/W16. W2 retains the proven provenance but lowers to gather/scatter
because same-binary measurement found no stable gain; W1 already uses scalar
memory. Permanent regressions cover aligned and row-crossing block geometry,
disabled lowering, W2 policy, a nine-element tail, a sparse cohort whose lane
zero index underflows, LLVM IR shape, final assembly without gather/scatter,
and independent numerical execution. The runtime reports accepted contiguous
read/write counts, and `LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER=1` is the A/B
oracle.

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

Direct texture reads and writes cross the JIT/runtime boundary once per
packet, not once per lane. The packet ABI carries the active lanes as low bits
of a `uint64_t`, coordinates as three `lane_count`-element SoA arrays, and a
pixel as four consecutive component vectors. The runtime may service a
same-texel broadcast once, batch a fully active contiguous row, or iterate only
set bits for a sparse cohort. It must inspect no inactive coordinate and issue
no inactive resource access; read scratch is initialized before the callback
so an inactive tail cannot expose poison.

The physical texture remains Luisa's native row-major resource layout. The
packet ABI is the SIMD-facing layout boundary: it permits coherence-aware
batching without changing native handles, upload/download layout, external
memory, or read/write synchronization. JIT code must not assume a raw texture
pointer or storage format. Direct wide AoS loads/gathers require a separate
proven safety and performance gate; the audited experiment reduced instruction
count but regressed end-to-end graphics time and is not part of production.

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
packed 64-bit sampler/base-extent field in a 16-byte descriptor total. Each
extent is limited to twenty bits, and a bindless update fails before publishing
the descriptor if an extent does not fit. JIT code issues exactly one callback
for a varying packet and
passes divergent slot indices through a SoA scratch array. Before any scratch
store, table lookup, coordinate conversion, mip conversion, or sampler decode,
inactive lanes are selected to benign zero operands. Callback result storage
is zero-initialized. The runtime then groups active lanes that resolve to the
same texture, sampler, and (where applicable) mip before accessing the native
row-major resource.

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
width return one without performing an invalid shift. Sampling without an
explicit level uses mip zero. A finite explicit level below zero uses zero;
finite positive levels clamp to the last allocated mip, `NaN` and negative
infinity use zero, and positive infinity uses the last allocated mip. Point
filtering retains the fallback contract and samples mip zero even when a
positive level is supplied. `LINEAR_POINT` selects the nearest mip while
`LINEAR_LINEAR` and anisotropic mode interpolate adjacent mips. A zero gradient
selects mip zero. A NaN in either derivative produces derived LOD zero before
an optional minimum-LOD clamp; a non-NaN infinite derivative selects the last
mip. The common 2D `BYTE1` stored-sampler path resolves the invariant
view once per packet and performs the four bilinear taps directly; other
formats and sparse masks retain the generic packet path.

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

Motion-instance resources accept two through `RTC_MAX_TIME_STEP_COUNT`
keyframes, a finite strictly increasing time range, and MATRIX or quaternion
SRT mode. A motion resource must be built before it is inserted into a TLAS.
The TLAS owns a copy of its 64-byte public keyframes and exposes only a stable
frame pointer, count, and mode through the plain instance table. MATRIX frames
are composed with the instance's outer affine before being supplied to Embree.
SRT frames map pivot/quaternion/scale/shear/translation to
`RTCQuaternionDecomposition`; Embree normalizes each nonzero quaternion and
performs quaternion interpolation. Since the deployed Embree ABI has only one
instance-stack level, an SRT motion instance currently requires an identity
outer affine; accepting a nonidentity affine would otherwise silently replace
quaternion interpolation with matrix interpolation. A time range beginning
inside the camera shutter requires `should_vanish_start`, and a range ending
inside it requires `should_vanish_end`, matching Embree's disappear-outside-
range behavior instead of inventing endpoint clamping.

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

Nonidentity outer affine composition for SRT motion,
`update_instance_buffer_only`, and deeper instance-stack behavior are not part
of this slice. They must fail at a specific capability boundary until their
independent semantic, IR-shape, and machine-boundary gates exist; triangle,
curve, procedural-query, and opacity support does not imply those deeper
instancing capabilities.

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
