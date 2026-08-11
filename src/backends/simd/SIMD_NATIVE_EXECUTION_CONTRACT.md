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

### 4.1 Width-one scalar CFG refinement

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
the packet return ABI remain shared with wider specializations. W4/W8/W16
continue to use the independent-lane cohort scheduler.

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

## 5. Vector-math providers

Provider selection is ordered:

```text
IR-native -> verified platform vector ABI -> unavailable
```

Warp-uniform operations use the scalar C/LLVM math operation exactly once.
A varying W1 specialization may use either one scalar operation or the shared
one-lane IR body. Varying `W = 4, 8, 16` operations must use a native
provider. They do not fall back to `W` scalar calls.

### 5.1 IR-native baseline

The portable baseline is implemented as LLVM fixed-vector algorithms. Range
reduction, polynomial/rational approximation, bit classification, table
gathers, and exceptional-value repair are expressed without target-specific
intrinsics. The same algorithm is instantiated for W4/W8/W16, then optimized
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

SIMD Schedule lowering instantiates these bodies at W1/W4/W8/W16. The
fallback backend uses the same provider for `sin`, `cos`, `tan`, `asin`,
`acos`, `atan`, `exp`, `exp2`, `exp10`, `log`, `log2`, and `log10` on DSL
float2/float3/float4 values, and uses its binary provider for `atan2`, so
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

Only FP contraction is permitted inside the fast bodies. They do not set
`nnan`, `ninf`, `nsz`, reassociation, approximate-function, or approximate-
reciprocal flags. This is what keeps the special-value contract meaningful.
Warp-uniform scalar expressions remain scalar and execute one LLVM/scalar
math operation; they are never splatted into the vector fast provider.

The regression instantiates precise and fast bodies at W2/W3/W4/W8/W16.
Every operation checks fixed boundaries and special values plus 8,192
deterministic raw float bit patterns, 8,192 domain-focused values, and 4,096
reduction/transition-focused values per width. The four independent
`exp2`/`exp10`/`log2`/`log10` bodies and binary `atan2` raise those three
deterministic corpora to 65,536, 65,536, and 16,384 values respectively.
The radix operations include integer and half-integer reduction points,
exponent transitions, and both mantissa partition boundaries; `atan2` uses
paired raw patterns plus ratio, quadrant, axis, infinity, and magnitude
partitions. Schedule tests cover W4/W8/W16 and inactive tails.
Optimized assembly is rejected if it contains a varying scalar libm symbol.
Power and hyperbolic functions remain explicit audit backlog and are not yet
marked SIMD-native by this checkpoint.

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
is not completion for a normal W4/W8/W16 path.

Embree traversal uses the packet API matching the specialization width:

| Width | Trace | Occlusion | Validity |
| ---: | --- | --- | --- |
| 1 | `rtcIntersect1` | `rtcOccluded1` | scalar active lane |
| 4 | `rtcIntersect4` | `rtcOccluded4` | 4-lane valid mask |
| 8 | `rtcIntersect8` | `rtcOccluded8` | 8-lane valid mask |
| 16 | `rtcIntersect16` | `rtcOccluded16` | 16-lane valid mask |

Ray/hit state is stored in packet-compatible structure-of-arrays form. The
current cohort mask and dispatch tail jointly form Embree's valid mask.
Inactive rays are initialized to benign values even though the validity mask
excludes them.

## 7. Executable audit matrix

Every newly supported device-library operation is accepted at three layers:

1. **semantic**: active lanes are compared with an independent scalar oracle,
   including domains, signed zero, infinities, NaNs, tails, and divergent
   masks;
2. **IR shape**: W4/W8/W16 contain fixed-vector operations or one approved
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
