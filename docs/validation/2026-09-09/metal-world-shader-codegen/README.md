# Metal world-shader code generation regressions

Validated on Apple M1 Max, macOS 26.6.2 (25G83), with Clang
21.1.8 and the Metal4 backend's LLVM 22.1.8. GPU tests and scene runs are
sequential. Both original-scene compile/dispatch gates completed.

The original failure uses Psycles `f690ead65ab5610a597b85b15e088ded70d0d540`
and its published Luisa gitlink `da8fff856066624bb56c4c24d06b1a6ffe108338`.
The fix starts from `4b0c02384` on `next`; the only intervening upstream
change is CI/xmake registration. No Psycles SVM or scheduler code is changed.

## Metal AST declaration ownership

`Local<T>` has function-allocated physical storage but fresh lexical epochs.
An exact whole-local `AssignStmt(local, UNDEFINED<T>())` marks each epoch.
It is not a zero fill, nor a proof that any component was initialized.

Metal's scope analysis may select the first dominating whole-local
assignment as the declaration site. When that assignment is the epoch seed,
the scope preamble intentionally emits no declaration. Dropping the seed
entirely then leaves every subsequent reference undeclared. The original
Lone Monk background SVM shader fails with undeclared `v0`, before coroutine
construction or renderer dispatch.

The correction preserves the declaration at the selected lexical site but
emits no initializer or assignment. If the seed was not selected as a
declaration site, the existing scope-owned declaration remains authoritative.
Ordinary `Var`/fixed-array zero initialization and other assignment paths are
unchanged. Programs still must initialize every component they read; no
promise about uninitialized bits or previous-loop contents is introduced.

`test_metal_codegen_regressions` now covers scalar, vector and array Local
representations, root/callable/branch/loop scopes, Local copying, dynamic
indices and repeated dispatches. Ordinary default-zero values are a negative
control. Every Local read is dominated by a program write. The original
backend fails at shader compilation; the source assertions additionally
require uninitialized declarations without synthetic zero fills. A BinaryIO
that always misses the cache permits checking the generated MSL.

## Metal4 optimizer-created constant storage

The original Metal4 background shader fails to link
`_switch.table.kernel_main` and `_switch.table.kernel_main_indirect`.
The minimal `test_metal4_switch_lookup` reproduces both failures using a
dense nonlinear 16-case integer switch. Its optimized IR contains a private
constant table in address space 0.

LLVM's switch-to-lookup optimization constructs its table with the module's
default global address space. See the
[LLVM 22.1.8 implementation](https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/llvm/lib/Transforms/Utils/SimplifyCFG.cpp#L6483).
AIR requires these immutable globals in constant address space 2, not its
ordinary thread-local address space. Before running the existing O2 pipeline,
the backend now supplies `G2` as the default global address space. It restores
the public data-layout spelling afterward so native-include ABI checks remain
unchanged. Existing globals already carry explicit address spaces; stack
allocations and threadgroup globals are not relocated. No optimization,
native math, or switch case is disabled. Compute cache revision is advanced
because the same AST/options now produce different AIR.

The new device regression checks every case and default through direct and
GPU-generated indirect dispatch, across three runtime input offsets and 257
threads. It uses uncached compilation. Both original AIR entries contain the
table, so a sibling kernel without an optimizer-generated global is not the
counterexample.

## Nullable Metal-cpp ownership

After the constant-table correction, successful pipeline loading hit a host
`brk` in `_load_kernels_from_library`, including for the ordinary Local kernel.
The shared Metal-cpp `NS::SharedPtr` implementation invokes C++ `retain` or
`release` through null pointers. Objective-C permits messages to nil, but the
preceding C++ member call still has undefined behavior. Clang 21 at O3 can
eliminate the optional-archive branch and terminate its success path with a
trap. Disassembly and the crash trace are retained separately from the
original unresolved-symbol failure.

The shared wrapper now explicitly guards nullable retain/release operations
in its helper, constructors, assignments, reset and destructor. Ownership
operations on nonnull pointees are unchanged. No backend compiler optimization
is disabled. `test_metal_shared_ptr` reproduces the failure without a GPU,
using an ordinary C++ refcounted pointee, and checks empty/converting owners,
copy/move counts and returning an aggregate with an empty optional member.
The original header traps; the correction passes 14 assertions in four tests.

## Focused validation

| Test | Backend | Result |
|---|---|---|
| `test_metal_shared_ptr` | host, O3 | 4 tests / 14 assertions pass |
| `test_metal_codegen_regressions` | Metal | 792 assertions pass |
| `test_counted_local_array` | Metal | 1028 assertions pass |
| `test_metal4_switch_lookup` | Metal4 | 1542 assertions pass |
| `test_metal_codegen_regressions --local-only` | Metal4 | 771 assertions pass |
| `test_counted_local_array` | Metal4 | 1028 assertions pass |
| `test_metal_xir_air` | Metal4 | 1108 assertions pass |

This is focused validation, not a full-suite pass. An additional attempt to
run the historical AST Metal fixture in full on Metal4 fails XIR verification
in its mutable-swizzle callable case (`metal4-legacy-swizzle-extra-test-failed.log`).
That fixture's existing assertions are not changed or counted as passing.
The new Local gate is independently selectable with `--local-only` and has
its own Metal4 CTest registration. The follow-up below resolves that historical
failure and adds the complete fixture to Metal4's CTest registration.

## Follow-up: callable swizzle references in AST-to-XIR

A multi-component swizzle lowers to an SSA `SHUFFLE` value. Passing it directly
to a callable's reference formal is invalid: the formal requires addressable
storage. This is a shared AST-to-XIR translator defect, not an AIR verifier
condition to waive. The original complete device fixture fails verification
before this repair. The initial minimal host test also fails six assertions
across direct and nested swizzles, including invalid-call and missing-writeback
checks (`callable-swizzle-red.log`).

The translator now captures the base address once, composes nested component
maps and gathers the selected components into a local temporary. It passes
that temporary by reference. For a writable formal, it scatters the resulting
components back immediately after the call and before any caller use of the
returned value. Readonly references have no copy-out. The scatter writes only
the selected components, not a stale copy of the whole parent vector. Ordinary
addressable references, resource/custom ABIs and external calls are unchanged.

For component map p, copy-in establishes T[i] = B[p[i]] at call entry. The
callee therefore sees the same swizzled value; copy-out establishes
B[p[i]] = T'[i], preserving all unselected components. The call result is
already an SSA value, so scattering before its caller use also preserves an
overlapping assignment such as `v.x = mutate(v.yx)`. Capturing the base address
before the call avoids re-evaluating dynamic array indices after callee effects.
This implements the existing callable swizzle copy-in/copy-out contract;
it does not promise pointer identity or general aliasing between temporaries.

The permanent `test_ast_callable_swizzle` now checks direct/nested swizzles,
dynamic array bases, reference operand types, exact component maps, ordering
relative to overlapping result stores and readonly negative controls: 48
assertions pass. The existing external-lowering fixture passes 20 assertions.
The complete unchanged device assertions pass on Metal (792) and Metal4 (775),
including texture and overlapping swizzle execution. Metal-only MSL source
checks account for the different counts. The test's optional argument guard
was also split into nested conditions: Boost.UT's overloaded `&&` otherwise
evaluated `argv[2]` even with only a backend argument. The no-option full run
now covers that guard as well. No expected device results were relaxed.
The 97 CFG tests / 2740 assertions were rerun successfully against this
translator build. Both complete original graph scene gates were then repeated
with clean production libraries: 1920x1080 / one sample, seven stages,
90 fields / 456 B, automatic tail and all 46 channels finite. Evidence is
under `callable-swizzle-canary/{metal,metal4}`. These gates do not replace
the subsequent 256-sample equal-pass performance campaign.

The original Lone Monk module is also exercised at 1920x1080, one sample,
using staged wavefront, independent direct-light queue and main shader cache
disabled. This is a compile/dispatch gate, not a performance comparison.
Both backends completed with 55 fields / 220 bytes, all 46 required channels
and zero nonfinite values. Metal used 131072 frame slots; Metal4 used 1048576.
This difference is the unchanged Psycles watchdog policy (`metal` is capped,
`metal4` is not). Their one-sample dispatch times are not extrapolated into a
256-sample speed comparison. The full 1080p/256-sample campaign follows SDK
publication, with session initialization separated from render-only timing.

## Evidence location

Local logs, original/fixed AIR, build configuration and full-scene artifacts:
`/Users/mike/CLionProjects/Psycles/build-macos/benchmarks/2026-09-09/lone-monk-metal-schedulers`.
Psycles disables dependency test targets in its embedded build. The local
`regressions/` harness compiles Luisa's test sources with the production
definitions and links the exact renderer libraries; permanent registration
also exists in Luisa's CMake/xmake build.
