---
name: spirv_codegen
description: Native Vulkan XIR-to-SPIR-V codegen, legalization, validation, bindings, control flow, and target features.
---

# Native XIR to SPIR-V

The native code generator lives in
`src/backends/common/spirv/spirv_codegen/` and lowers XIR to SPIR-V 1.5 with
glslang's `spv::Builder`. Vulkan enables it with
`LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV` (CMake, default ON) or
`lc_vk_backend_use_xir_spirv` (xmake, default true). The two Vulkan codegen
routes are mutually exclusive: enabling this route together with
`LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV` (CMake, default OFF) /
`lc_vk_backend_use_ast_llvm_spirv` (xmake, default false) fails at configure
time (`FATAL_ERROR` in `src/backends/CMakeLists.txt` and the root
`CMakeLists.txt`; `raise()` in `scripts/xmake_func.lua`, which also drops the
default XIR value when only the AST route was requested). The selected route
defines `LUISA_XIR_TO_SPIRV` or `LUISA_AST_LLVM_TO_SPIRV` on the Vulkan
backend target.

This skill describes the native path, not the separate
AST -> LLVM -> SPIR-V implementation in `spirv_llvm/`.

## Source map

| Files | Responsibility |
|---|---|
| `entry.h/.cpp` | Public compile entries, target-feature state, result assembly, validation and optimization |
| `utils.h/.cpp` | AST -> XIR and the backend legalization pipeline |
| `dialect.h/.cpp` | Fail-closed XIR handoff validation |
| `ray_query_lifetime.h/.cpp` | `validate_spirv_ray_query_lifetimes` opaque-value lifetime and argument rules |
| `pointer_legalization.h/.cpp` | SPIR-V callable ABI specialization policy |
| `argument_usage.h/.cpp` | Fixed-point function-argument usage shared by legalization and emission |
| `bindless_usage.h/.cpp` | Exact global-heap and per-array metadata requirements by resource opcode |
| `texture_sampling.h/.cpp` | Canonical sampler selector, image dimensionality, and target-contract planning |
| `structural_closure.h/.cpp` | Canonical backend-emitted block closure and malformed-role diagnostics |
| `call_graph_validation.h/.cpp` | Kernel-reachable, callee-before-caller function graph |
| `control_flow_plan.h/.cpp` | Immutable logical-to-physical structured-CFG plan |
| `instruction_layout.h/.cpp` | SPIR-V instruction word-count limits, including `OpSwitch` and `OpPhi` |
| `buffer_layout.h/.cpp` | Vulkan typed-SSBO layout compatibility and word-storage fallback planning |
| `atomic_buffer_plan.h/.cpp` | `plan_spirv_atomic_buffers` single typed-vs-word representation per `Buffer<T>` |
| `constant_ubo_layout.h` | Checked std140 constant-UBO member planning and the portable 16 KiB limit |
| `kernel_argument_layout.h/.cpp` | `plan_spirv_kernel_argument_layout` checked argument-block placement over the common `ArgumentBlockLayout` |
| `aggregate_index.h/.cpp` | Typed GEP/extract/insert index planning and struct-index canonicalization |
| `arithmetic_support.h` | SPV_EXT_float8 permitted-op subset and GLSL.std.450 transcendental width contracts |
| `atomic_target_contract.h/.cpp` | `validate_spirv_atomic_target_contract` atomic vs enabled-target validation |
| `optimizer.h/.cpp` | SPIRV-Tools validation and optimization presets |
| `target_feature_mask.h` | Persisted required-feature bit contract |
| `target_features.h` | Logical-device feature snapshot and pure lowering decisions |
| `runtime_target_plan.h/.cpp` | Pre-binding descriptor, ray-query, subgroup, and bindless runtime contract |
| `kernel_argument_role.h` | Stable per-argument native runtime role bits (accel traversal/instance, buffer/bindless device-address) |
| `src/backends/vk/shader_artifact_codec.h/.cpp` | Canonical Vulkan shader-artifact writer/parser, integrity checks, SPIR-V validation and feature reconciliation |
| `property.h` | `hlsl::ShaderVariableType`/`hlsl::Property` aliases used by the descriptor ABI |
| `bind.cpp` | Descriptor properties, argument buffer, bindless heaps, constant UBO |
| `type.cpp` | Logical and storage-layout type conversion |
| `emit.cpp` | Module/function/block emission, prologues, native Phi emission |
| `control_flow_optimization.h/.cpp` | Backend-only loop-control literal planner (`Unroll` proof) frozen beside the CFG plan |
| `memory_access_plan.h/.cpp` | Exact `Aligned` operand proof for uint32 word-ABI loads/stores |
| `condition_inst.cpp` | Emission from the frozen control-flow plan |
| `instruction.cpp` | Non-control instructions, resources, atomics, ray query, builtins |

CMake glob-registers `spirv_codegen/*.cpp` in `luisa-compute-spirv`; xmake
does the same for `lc-spirv`.

## Compilation contract

There are two supported entries:

```cpp
SpirvResult compile_spirv(Function kernel, const ShaderOption &,
                          SpirvTargetFeatures);
SpirvResult compile_spirv_xir(Function kernel, const xir::Module *,
                              const ShaderOption &,
                              SpirvTargetFeatures);
```

`compile_spirv` translates and legalizes AST input. `compile_spirv_xir` is for
an already legalized module and is used by exact-XIR backend tests. The AST
`Function` remains the external descriptor/argument ABI in both cases.

The compile sequence is:

1. Validate the AST/XIR kernel ABI and the native SPIR-V XIR dialect. Generic
   XIR validity remains whole-module; backend-specific work uses the canonical
   kernel-reachable call graph and each reachable function's structural
   closure.
2. Freeze the reachable functions, used types/constants, exact bindless usage,
   atomic-buffer representation, atomic target contract, sampler contract, and
   runtime-only target-feature plan before descriptor or instruction emission.
3. Compute fixed-point argument usage and exact per-argument acceleration and
   bindless-metadata roles. Merge only the public synchronization usage that is
   deliberately conservative; descriptor roles remain optimized-XIR exact.
4. Generate descriptor properties, the internal argument block, global heaps,
   and per-argument resource globals from those immutable plans.
5. Emit reachable functions through immutable control-flow plans.
6. Dump the module and validate it for `SPV_ENV_VULKAN_1_2`.
7. Run the selected SPIRV-Tools optimizer preset and commit its output only if
   optimization and validation both succeed.
8. Validate the selected final binary, reconcile capability-owned target bits,
   then return the exact properties, argument usage/roles, and feature mask
   that serialization and runtime binding consume.

The pre-optimization module must be valid. The optimizer is never a repair
step for malformed SPIR-V.

The same rule applies inside emission: generic XIR, the native dialect, and
the frozen AST/XIR ABI are hard invariants. Do not coerce invalid arithmetic
operands or synthesize missing kernel arguments. Assert if a verified boolean
`select` or planned argument buffer does not lower as expected.
`_ensure_type` is an internal same-shape numeric conversion helper, not a
general cast operation: boolean conversion belongs to XIR `STATIC_CAST`, and
unsupported class or shape pairs must fail rather than fall through to
`OpBitcast`.

Emit ordinary XIR arithmetic as ordinary SPIR-V instructions. Do not turn
constant operands into `OpSpecConstantOp`: the runtime exposes no matching
specialization-constant ABI. Keep optional integer constant folding and
strength reduction in the XIR/SPIRV-Tools optimization layers instead of
embedding partial, always-on peepholes in instruction selection.

`SpirvCodegenEntry` owns its `spv::Builder` normally. Its destructor first
clears maps keyed by XIR objects, then the builder is destroyed. Do not add a
`release()` leak workaround.

## AST to XIR legalization

The mandatory final pipeline in `utils.cpp` establishes backend semantics even
when optional optimization is disabled:

1. Lower ray-query loops to ordinary loop form.
2. Promote safe read-only reference arguments.
3. Optionally optimize still-structured XIR.
4. Run `destructure_cfg`.
5. Run SPIR-V pointer/resource-call specialization.
6. Optionally run ordinary inlining and scalar/SSA cleanup.
7. Temporarily run `reg2mem` because `restructure_cfg` currently requires
   Phi-free raw CFG.
8. Run `restructure_cfg`.
9. Clear payload only from blocks outside ordinary all-edge reachability and
   immediately run `mem2reg`, then audit that no typed reg2mem spill provenance
   remains; retain every block identity.
10. Fix self-referential values and validate the dialect.

The temporary `reg2mem` is a boundary adapter for restructuring, not the
codegen representation. Do not add a final blanket Phi elimination. Native
`OpPhi` emission owns reconstructed SSA and avoids imposing avoidable memory
traffic on the generated module or on `spirv-opt`.

Every temporary slot created by generic `reg2mem` carries
`Reg2MemSpillMD` with `PHI` or `CROSS_BLOCK` provenance. The SSA transport slots
created by `restructure_cfg` across cloned-region boundaries also carry
`CROSS_BLOCK`: their diagnostic comments must not pin them in memory after the
final `mem2reg`. Preserve this typed metadata through instruction cloning,
SROA splitting, and XIR text/bitcode
round-trips; names and comments are diagnostic only. Ordinary final
legalization recovers SSA immediately after restructuring. Pre-autodiff
legalization is the intentional exception: autodiff requires Phi-free IR, so
typed spills remain in memory form across autodiff and SROA. The final
post-restructure audit then examines every metadata owner and fails if any
marker remains or is misplaced. The exact-XIR dialect independently rejects a
tagged alloca in its active structural closure. Untagged user local allocas
remain a supported XIR/SPIR-V construct and must not be rejected by this
boundary.

Generic multi-block inlining must run after `destructure_cfg`. A target-specific
exception belongs in `pointer_legalization.cpp`, which uses the generic atomic
`inline_call_sites_pass_run_on_module` primitive only after preflighting the
whole selected batch.

The legalization `inline` step opts in to
`InlineOptions::preserve_invocation_convergence` (`utils.cpp`). Because this
route destructures first and restructures afterwards, cloning a callee that can
exit early would make the caller's continuation conditional per invocation, and
`restructure_cfg` could only rebuild it as a guarded region; subgroup
collectives, block/split barriers, quad derivatives, and `SHADER_EXECUTION_REORDER`
reached directly or transitively in the caller would then execute with a
partially active scope. Both `inline_all` and `inline_call_sites_pass_run_on_module`
keep such calls intact and report `skipped_convergence_call`. The option is a
backend policy passed explicitly by this pipeline, not a change to the generic
inliner default, and `test_xir_passes` plus `test_fp4_quantization vk` pin it.

## Dialect boundary

`validate_spirv_xir_codegen_dialect` is the single fail-closed handoff. When
adding an XIR opcode or type:

- classify it explicitly as supported, semantic no-op, or unsupported;
- validate operand count, types, storage class and backend semantic limits;
- add a focused diagnostic rather than relying on an emitter assertion;
- update the complete opcode-matrix test;
- add an exact codegen test when the construct is accepted.

Keep two scopes deliberately separate. Generic `xir_verify_module` validity is
a whole-module contract, including unused definitions and orphan blocks. The
native dialect, target/resource planners, and emitter consume only the
canonical kernel-reachable function graph and each function's structural
closure. Never weaken generic verification to reachable-only, and never let an
unused callable or true orphan manufacture descriptors, target features, or
emitted SPIR-V.

Important type boundaries include:

- `OpTypeArray` length is strictly positive; XIR may still represent a
  zero-length host type, so the SPIR-V dialect rejects it;
- storage `ArrayStride` is strictly positive; buffers with zero-sized elements
  and nested storage arrays with zero-sized elements are rejected;
- `SPV_EXT_float8` permits FP8 values only in its listed transport, storage,
  conversion, composite and selection instructions. General FP8 arithmetic
  and comparisons remain invalid. Do not classify `OpTranspose` as transport:
  it belongs to the matrix-instruction category, which the extension does not
  admit. FP8-to-bool casts must widen each scalar lane to float32 before the
  unordered comparison so NaN remains truthy;
- texture dimensions are 2 or 3 and scalar elements are float32/int32/uint32;
- texture writes consume exactly the verifier-mandated four-component texel
  vector. The emitter must not pad, smear, or otherwise repair a scalar or
  short-vector payload that crossed the dialect boundary;
- opaque ray-query values have deliberately restricted lifetime and argument
  rules;
- kernel reference arguments are not part of the Vulkan descriptor ABI.

### Structural closure and inactive payload

All backend analyses and emission use
`plan_spirv_codegen_structural_closure`. It contains the ordinary CFG reachable
from the function body plus every raw structured role block recursively owned
by that CFG. Function-owned blocks outside this set are true orphans and do not
participate in exact native emission, descriptor analysis, uniformity, or
planner ownership.

A raw role block can belong to the structural closure without being ordinarily
reachable. Exact XIR emission accepts such a disconnected role payload only
when it has a self-contained, flat value/lifetime contract: no Phi, nested
structured owner, Break/Continue, opaque ray-query state, cross-block
instruction value, forward same-block use, or branch re-entry. Its terminator
must be Return or Unreachable. The emitter and resource/callable analyses must
still inspect accepted payload because direct exact-XIR codegen has not run a
dead-payload pass.

glslang's mandatory `postProcess(false)` subsequently rewrites an
ordinary-unreachable merge or continue target to its canonical unreachable
form, so instructions from that payload do not appear in the dumped SPIR-V.
Boundary tests must prove structural closure, callable discovery, and resource
planning before this canonicalization, then expect the dead payload opcodes to
be absent from the final validated binary.

After `restructure_cfg`, every executable transfer is explicit.
`clear_spirv_codegen_inactive_block_payloads` may therefore replace payload in
both true orphans and ordinary-unreachable role blocks with Unreachable before
`mem2reg`. These categories remain distinct: a true orphan is outside the
exact-emission closure, while a disconnected role keeps its structural block
identity even when this mandatory legalization proves its payload dead. Do not
make optional DCE determine closure membership or merge ownership.

## Callable pointer and resource legalization

SPIR-V policy must not leak into the generic XIR inliner. The backend computes
fixed-point argument usage, then specializes only call sites whose retained
callable ABI cannot be represented safely.

Current rules:

- a reference formal receiving a function-local `AllocaInst` or compatible
  reference argument may remain a Function pointer;
- a shared alloca is Workgroup storage and must be specialized;
- indirect-dispatch buffer arguments are always specialized;
- a resource formal with a proven unique kernel-resource origin may remain
  outlined; emission recovers the descriptor binding and omits that formal
  from the SPIR-V function ABI;
- otherwise, used buffer and bindless resource formals are specialized into
  the call site;
- without a unique kernel-resource origin, a writable acceleration structure
  or a texture used for both read and write is specialized;
- a genuinely unused resource formal need not force specialization.

Legalization is fixed-point because inlining one layer can expose a pointer at
another layer. It must preflight recursion, call shape, structured boundaries,
and all selected sites before mutating any function. Ordinary switches that do
not block a selected inline remain native switches.

Do not solve callable ABI failures by enabling `VariablePointers` globally.
Descriptor-backed buffer/bindless arguments either resolve to proven global
bindings or are specialized; they do not become storage-buffer pointer
parameters. Only safe opaque/resource modes remain as callable parameters.

## Structured control flow and Phi

`ControlFlowPlan` freezes the final physical graph before instruction emission.
It owns:

- reverse-post-order block schedule;
- construct headers, merge targets and continue targets;
- synthetic loop headers, continues and edge trampolines;
- source-sensitive merge routing;
- physical loop-boundary validation;
- logical Phi incoming paths through synthetic forwarding blocks.

Emission preallocates every physical block. It then predeclares one `OpPhi` in
the result block and, where necessary, auxiliary `OpPhi` nodes in forwarding
blocks. Incoming values are resolved at the logical predecessor tail before
its terminator. Finalization checks that planned and actual physical
predecessor sets are identical.

Rules:

- the physical SPIR-V function entry has no predecessor and no Phi;
- `Loop.prepare` has exactly one non-null `BasicBlock` role operand and has
  exactly one of two terminator forms: `Branch(Loop.body)` or
  `ConditionalBranch(bool, Loop.body, Loop.merge)`; both are native loop
  headers and lower directly to `OpLoopMerge` followed by the matching branch;
- an `OpPhi` must be the first non-line instruction in its block;
- every physical predecessor appears exactly once;
- a loop header has one entry and one backedge, and the backedge passes through
  the declared continue target;
- one SPIR-V merge block cannot be owned by multiple constructs;
- forwarding a logical incoming through a synthetic edge requires a Phi in
  that forwarding block, not reuse of a non-dominating value;
- use `plan_spirv_phi_instruction` and `plan_spirv_switch_instruction` before
  allocating variable-length instructions; SPIR-V word count is 16-bit.

Never rediscover or redirect edges during emission. Add facts to
`ControlFlowPlan`, validate them there, then consume the frozen plan.

## Backend-only emission policies

These decisions live entirely in this directory (plus the Vulkan target-feature
and artifact contract where a capability is involved). None of them may add an
XIR operation, metadata, or hint.

| Emission | Policy / proof | Fallback |
|---|---|---|
| `xir::ArithmeticOp::ROUND` | `SpirvRoundPolicy` (see "Floating-point contraction" above) | portable sequence |
| `WARP_READ_FIRST_ACTIVE_LANE` | one `OpGroupNonUniformBroadcastFirst` per scalar/vector value, one per matrix column; requires ballot only | n/a (no capability is gained or lost) |
| `WARP_READ_LANE` | one `OpGroupNonUniformShuffle` per scalar/vector value, one per matrix column; still requires `GroupNonUniformShuffle` | n/a |
| Splat composites | `setUseReplicatedComposites(true)` only when the device negotiated `shaderReplicatedComposites` | ordinary `Op*Composite` |
| Dispatch-bounds predicate | `OpExpectKHR(pred, false)` only when `shaderExpectAssume` is enabled | plain predicate |
| Loop-control literals | `ControlFlowOptimizationPlan`: `Unroll` only for an exact constant trip count, a provably duplicable payload, and a checked code-size budget | `MaskNone` |
| Word-ABI memory operands | `plan_spirv_aligned_word_access`: `Aligned 4` only when the existing byte-offset proof is exactly ≥ 4 | original masked access |

Subgroup reads: SPIR-V admits only scalar/vector result types, so a matrix is
decomposed into column vectors, read one column at a time, and reconstructed.
Never pass an `OpTypeMatrix` to `OpGroupNonUniformBroadcastFirst` or
`OpGroupNonUniformShuffle`, and never scalarize a vector into one read per
component. `OpGroupNonUniformBroadcastFirst` is exactly "the value of the first
active invocation", so it replaces `Ballot` + `BallotFindLSB` + one shuffle per
component without changing semantics and without requiring the shuffle
capability: a first-active-only module needs `GroupNonUniform` +
`GroupNonUniformBallot` (`+ GroupNonUniformExtendedTypes` for narrow types) and
nothing else.

`SPV_EXT_replicated_composites` (not `SPV_KHR_...`) is negotiated from
`VK_EXT_shader_replicated_composites` separately from
`VK_NV_cooperative_vector`, which merely *requires* it. A cooperative-vector
module therefore also requires the replicated-composite feature: glslang builds
a cooperative-vector splat with `OpCompositeConstructReplicateEXT` on the NV
type class regardless of the builder flag, so any fixture or artifact that
enables `cooperative_vector` must enable `shader_replicated_composites` too. glslang declares
`ReplicatedCompositesEXT` itself only when it really emits a replicated
instruction, so the persisted bit is capability-reconciled and a module without
one never claims it; the capability is deliberately outside
`can_safely_trim_capabilities`, which leaves such modules untrimmed. The
extension is negotiated in `device.cpp` rather than through
`OptionalExtensionRegistry` (which would chain a second
`VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT`), so it is intentionally
absent from `VulkanExtensionStatusExt`.

The dispatch-bounds branch is backend-owned, so `OpExpectKHR` needs no XIR
likelihood hint. It changes optimization metadata only: the compare, the branch
targets, and which invocations return are untouched. `OpAssumeTrueKHR`,
`SubgroupUniformControlFlowKHR`, `Nontemporal`, and any XIR-carried hint stay
out of scope.

`ControlFlowOptimizationPlan` is built from the same immutable XIR as
`ControlFlowPlan`, before any physical block ID exists, and emission only
consumes it. `LUISA_SPIRV_LOOP_HINTS=0` disables it and the environment value is
part of the shader cache identity. Synthetic software loops (compare-exchange
remainder, ray-query traversal) stay `MaskNone`; only `xir::LoopInst` regions are
candidates, and any unknown trip count, early exit, nested construct, memory
access, atomic, thread-group operation, call, or over-budget expansion rejects
the loop. `SelectionControlMask::Flatten`, `DontUnroll`, dependency lengths,
peel/partial counts, and min/max-iteration hints have no exact proof in the
current XIR and must not be invented.

`plan_spirv_aligned_word_access` annotates only ordinary uint32 word-ABI
loads/stores, never atomics, the masked compare-exchange window, typed or image
accesses, `NonPrivatePointerKHR`, or `Nontemporal`. It never rounds a proof up:
one- and two-byte proofs stay unannotated, and every existing memory-access bit
(notably `Volatile`) is preserved rather than replaced.

## Types and layout literals

`_convert_type` handles logical SSA types. `_convert_laid_out_type` recursively
decorates buffer payloads with `Offset`, `ArrayStride`, `ColMajor`, and
`MatrixStride`.

SPIR-V layout and binding literals are unsigned 32-bit words. glslang's
single-literal convenience overload takes `int`; do not route a wide ABI value
through it. Use the `vector<unsigned>` decoration overload for variable
offsets, strides, descriptor sets, and bindings. For `makeArrayType`, a small
nonzero third argument may mark the type explicitly laid out, followed by the
real unsigned `ArrayStride` decoration.

Logical bool has no StorageBuffer representation. A second mismatch comes from
64-bit three- and four-component vectors: Vulkan gives them 32-byte standard
storage alignment, while Luisa host vector alignment is capped at 16 bytes.
`plan_spirv_typed_buffer_layout` recursively checks matrix/array strides,
structure member offsets, structure stride, and the outer runtime-array stride.
Any incompatible non-atomic `Buffer<T>` uses the byte-exact uint32 word ABI.
Atomic analysis consumes the same layout decision and selects one
representation per `Buffer<T>` before type conversion; a 64-bit integer atomic
that requires typed storage conflicts with a layout that requires word storage
and must fail at the handoff. Never enable scalar-block layout implicitly: the
runtime does not request that Vulkan feature as part of this ABI.

Direct-buffer `StorageBufferMetadata` carries a runtime descriptor bias, but
the Vulkan argument preprocessor proves that a typed buffer view's offset and
size are exact multiples of its logical element stride. Preserve the resulting
`gcd(element_size, 4)` alignment in word-storage reads and writes. Dropping it
to one byte needlessly expands ordinary aligned stores into masked atomic-CAS
loops. Raw byte-buffer operations have no such proof and remain alignment one.

XIR atomics specify atomicity but expose no memory-order operand. Emit SPIR-V
atomics with `Relaxed` memory semantics while retaining the pointer-derived
Device or Workgroup scope. This matches CUDA/HIP `Monotonic` and fallback
`__ATOMIC_RELAXED`; block synchronization and runtime resource barriers own
visibility ordering. Do not attach `AcquireRelease` or broad memory-class bits
to every RMW: that silently strengthens the cross-backend contract and can
serialize unrelated atomics. Compare-exchange success and failure semantics,
including software float CAS loops, are both relaxed.

Large array constants may use the portable constant UBO planner. Only layouts
with an exact host-to-std140 serializer are eligible. Planning is checked for
alignment, multiplication, cumulative range, and the portable 16 KiB limit.

## Aggregate indices

All GEP, dynamic extract/insert, and atomic address walks use
`plan_spirv_aggregate_indices`.

- array/vector/matrix/buffer indices retain their legal integer value IDs;
- structure indices must be constant and are canonicalized to unsigned 32-bit
  `OpConstant` IDs;
- planning validates the whole walk before emission;
- usage analysis must track the canonical emitted index, not keep an otherwise
  dead narrow or 64-bit source constant alive.

## Bindings and Vulkan ABI

`hlsl::Property` is a persisted ABI shared by codegen, serialization, Vulkan
layout creation, and dispatch binding. The backend uses:

- set 0: dense local descriptors;
- set 1: sixteen immutable samplers;
- sets 2+: enabled buffer, 2D texture, and 3D texture heaps in that order;
- `ConstantValue`: a descriptor-free push-constant pseudo-property.

All consumers must ignore `ConstantValue` during descriptor lookup regardless
of property order. Writer and reader both validate the canonical property
shape. Runtime planning separately checks ordinary descriptor limits,
update-after-bind aggregate limits, acceleration-structure limits, set count,
and `vkGetDescriptorSetLayoutSupport`.

Keep internal emission roles explicit beside each public property. Debug names
such as `_Global`, `_bdarr_*`, `tex2d_heap`, and `tex3d_heap` may describe the
generated module, but must not select argument-buffer IDs, bindless table
types, texture dimensions, or heap IDs. Unknown public property kinds fail
closed before a `NoResult` ID can enter the interface list.

Native direct-buffer views use an internal argument buffer. Non-resource
values are host-layout packed first; `StorageBufferMetadata` records follow at
their natural alignment. HLSL debug validation words use a mutually exclusive
trailer. The common checked argument-block planner is the source of truth for
both sizing and emission.

Storage-buffer alias decorations describe backing memory, not merely one
descriptor access path. Luisa permits the same buffer or overlapping views to
be supplied through multiple arguments, including imported native resources
and bindless arrays. Emit `Aliased` on every user-bindable storage-buffer
declaration that may participate. A read-only declaration may carry
`NonWritable` only when the module has no writable user-resource path that
could alias its backing memory; include direct buffers, bindless buffer stores,
writable accel-instance storage, writable textures/external memory, and the
custom indirect-dispatch buffer in that proof. Do not infer immutability from
the declaration's local `Usage::READ` alone.

Volatile direct-buffer accesses require three matching SPIR-V facts: a
`Volatile` memory operand on each load/store, the backend's matching device
fence, and `Coherent` on that exact buffer declaration. Propagate coherence as
an exact fixed-point argument role through callables; do not mark every buffer
with the same element type coherent, because that needlessly disables caching
for unrelated resources. `Coherent` is not a substitute for an uncertain alias
contract: omit `NonWritable` when a read declaration may alias writable user
memory, but do not make it coherent. The backend-owned `_Global` argument block
and bindless metadata blocks cannot alias user resources; keep them
`NonWritable` and do not decorate them `Aliased` or `Coherent`.

Ordinary XIR `LoadInst` and `StoreInst` are exact-typed memory operations: the
address is an lvalue of the loaded/stored type, and a stored value is an
rvalue. Enforce that contract at the dialect handoff. Do not smear scalars or
insert bitcasts in the SPIR-V emitter to make a mismatched store validate;
those conversions manufacture semantics for invalid XIR and can hide an
upstream pass defect.

Bindless buffer planning has two independent facts. A real bindless read/write
needs the global unbounded buffer heap and the matching array's local metadata
descriptor; a size-only query needs only that local metadata descriptor.
`SpirvResult::useBufferBindless` therefore means the global heap only. Fixed-
point argument analysis emits `SPIRVBindlessBufferMetadata` only beside each
bindless argument that actually needs it; do not turn this back into one
module-wide optional descriptor per bindless argument.

The native route currently accepts only ordinary `BindlessSlotType::MULTIPLE` layout
operations whose uniformity can be proven from XIR. Typed and explicit
uniform-index AST operations remain honest HLSL fallback reasons: XIR resource
instructions do not yet preserve the typed slot layout or the caller's
uniform-index promise. Do not erase those route guards or map typed operations
onto ordinary resource ops. Native support requires first-class orthogonal
layout/index-mode flags through AST↔XIR, cloning, verification, text/bitcode,
callable argument analysis, persisted Vulkan argument roles, runtime layout
checking, and SPIR-V slot resolution before the fallback can be relaxed.

For a divergent descriptor lookup, apply `NonUniformEXT` to the actual
descriptor-array index and preserve it through the resulting access-chain
pointer, descriptor load, and consuming image/sampler value as required. Do
not decorate prefix structure/array indices. Those are commonly interned
constants such as zero; decorating one contaminates every unrelated use of the
same module-global SPIR-V ID and may unnecessarily pessimize driver analysis.

Acceleration structures likewise have two independent native roles: traversal
uses `SPIRVAccel`, while instance-property access uses the separate
`SPIRVAccelInstance`/`SPIRVAccelInstanceRW` buffer. `SpirvResult::argument_roles`
is parallel to `argument_usages` and persists the exact role mask in
`SavedArgument::resource_aux`. A zero role mask is valid for an unused native
accel. The all-ones sentinel means an older/non-native artifact whose role is
unspecified; native serialization and dispatch must never infer optional accel
descriptors greedily from neighboring properties.

### Ray-query traversal

`OpRayQueryProceedKHR` advances traversal; a true result means traversal is
still incomplete. Direct closest-hit tracing must therefore emit a structured
loop that calls `OpRayQueryProceedKHR` until it returns false before reading
committed intersection fields. The `spv::RayFlagsMask::OpaqueKHR` flag (emitted
with `SkipAABBsKHR` for the direct closest-hit query) removes
candidate-intersection handling, but it does not make one call sufficient.
Direct any-hit tracing adds `TerminateOnFirstHitKHR`, so its single proceed
call remains intentional.

Keep both sides covered: a structural SPIR-V test should distinguish the
closest-hit loop from the any-hit single call, and a Vulkan runtime test should
place a farther primitive before an overlapping nearer primitive so a
premature committed read cannot accidentally pass.

Persisted shaders include section sizes and hashes for properties, saved
arguments, SPIR-V, printers, and constant data. `require_recompile` must parse
and validate the complete artifact; a matching prefix is not sufficient.

`shader_artifact_codec` is the single production owner of the compute and
raster artifact format. Fresh serialization, `require_recompile`, and live
shader loading must all use its encoder/decoder rather than maintaining
parallel header or section parsers. The decoder verifies the semantic header,
bounded total size, every section digest, the persisted interface and printer
records, and the stage-specific `main` entry point of every module; only
`XIR_SPIRV` modules additionally run SPIRV-Tools validation for
`SPV_ENV_VULKAN_1_2` (DXC-produced HLSL artifacts may carry SPIR-V beyond
that profile) before returning decoded data. Printer records use a bounded, non-fatal parser
for the narrower `ShaderPrintFormatter` type and brace dialect; artifact text
must never be passed directly to the fatal `Type::from` parser. The codec
applies final-capability reconciliation only to `XIR_SPIRV`; HLSL and LLVM
artifacts retain their independently produced feature contracts. Tests should
round-trip the production codec through an in-memory `BinaryIO` and recompute
hashes in malformed-SPIR-V and printer fixtures so they cannot pass by
exercising only framing helpers.

## Target features

Every optional capability has two sides:

1. codegen records the exact logical-device feature needed by the emitted
   artifact and rejects it when that feature is unavailable;
2. the resulting required-feature mask is persisted and checked against the
   enabled logical-device mask on load.

Operations call `_require_target_feature(bit, enabled)` at the semantic
emission site to record a provisional requirement. Availability validation is
deferred until after optimization only for one-to-one capability-owned bits:
dead-code elimination may remove the last feature-bearing instruction, and
the final SPIRV-Tools trim-capabilities pass may then remove its declaration.
Runtime/layout and lowering-owned bits still fail immediately. Codegen
reconciles capability-backed bits from the final validated binary before
checking the complete logical-device mask and persisting the artifact. This is
also how narrow arithmetic remains distinct from narrow storage: Vulkan's
8/16-bit storage capabilities permit restricted loads, stores, and width-only
conversions without `Int8`, `Int16`, or `Float16`, while constants and
arithmetic retain the shader capability.

Only one-to-one `OpCapability` requirements are final-binary-owned. Runtime
layout and semantic features that cannot be reconstructed from capability
declarations—descriptor binding flags, sampler anisotropy, storage-class-
specific atomics, subgroup extended types, and ray-query descriptor ABI—stay
emission-owned. Never clear them merely because a similarly named capability
is absent. Conversely, do not strip unsupported capabilities with a raw
binary rewrite. The vendored SPIRV-Tools trim pass has an explicit, incomplete
input contract, so codegen registers it only when every declared capability is
in the locally audited allowlist. Storage-only
`UniformAndStorageBuffer16BitAccess` without `Int16` or `Float16` is excluded
because this pass revision would remove that live capability. Outside the
audited domain, retain provisional requirements conservatively.

Final reconciliation does not make every dead unsupported source operation
compilable. Feature-dependent planners still run before SPIR-V exists:
buffer float atomics may choose a word/CAS fallback, shared float atomics may
reject when no legal fallback exists, sampler anisotropy is validated against
the runtime sampler heap contract, and narrow constant UBO lowering is selected
only when the matching Uniform-storage feature is enabled. Tests at the final
artifact boundary must start from an already legal emitted module and must not
imply otherwise.

Vulkan device creation must preserve the same independence. In particular,
`shaderFloat16`, `shaderInt8`, `storageBuffer8BitAccess`,
`uniformAndStorageBuffer8BitAccess`, `storageBuffer16BitAccess`, and
`uniformAndStorageBuffer16BitAccess` are six separate feature bits. Never
enable or advertise them as an all-or-nothing width bundle.

Unknown persisted bits fail closed. An imported `VkDevice` does not reveal
which optional features were enabled, so the backend must not infer enabled
features from physical-device support. Bindless update-after-bind is disabled
for imported devices without an explicit feature contract.

Cache and AOT loading follows a strict integrity order. The fixed-width header
has a canonical semantic digest built by appending each field in little-endian
order; never hash raw struct padding. Verify that digest before trusting the
persisted dialect or feature mask. Then bound all serialized section sizes,
read the complete payload, verify every section digest, and validate SPIR-V for
the Vulkan environment before interpreting final capabilities. For the
serialized `XIR_SPIRV` dialect only, reconcile all capability-owned bits from
the validated module (or the union of stage modules) and require an exact match
with the persisted mask. Preserve emission-owned bits from the integrity-
protected header, then check the reconciled requirements against the logical
device. Do not apply native capability-accounting assumptions to HLSL or LLVM
artifacts. Builds with the LLVM SPIR-V path still link the common validator and
validate loaded modules before pipeline creation.

Cache consumers also state their required shader identity explicitly. Compute
cache reads require shader MD5, type MD5, and the selected codegen dialect on
both preflight and actual deserialization; native JIT uses `XIR_SPIRV`, while
HLSL compilation states its own dialect. Raster AOT loading requires the
expected type MD5 and `HLSL_SPIRV` dialect during codec decoding, before any
Vulkan pipeline is constructed. Generic AOT loading leaves a dimension
unconstrained only through an explicit load-requirements field. This prevents
a matching filename or shader hash from crossing a codegen-dialect or
argument-type boundary.

When adding a feature, update:

- `SpirvTargetFeatures` and `target_feature_mask.h`;
- Vulkan physical support query and logical feature enabling;
- emitted extension/capability;
- serialized required-feature mask and pipeline-environment hash;
- exact supported, missing, and cache-load tests.

## Constants and exact bits

Floating constants use their exact XIR bit representation. The local glslang
extension `makeFpConstantFromBits` interns the declared-width payload directly;
do not round-trip through host `double`. Tests cover signed zero, NaNs and
payload preservation.

Use glslang's width-matched integer constant constructor. A 64-bit
`OpConstant` requires two literal words, even when its value is only zero or
one; `makeIntConstant(type, value, ...)` emits one word, so use
`makeInt64Constant` for every 64-bit integer type. Keep a SPIRV-Tools
validation fixture for helper-generated constants such as integer/bool cast
zero and one values.

Variable-length literals such as switch cases retain the selector width and
signedness in XIR/text/binary interchange. SPIR-V `OpSwitch` emits the exact
one- or two-word case encoding required by the selector width.

## Validation and optimization

`validate_spirv` uses SPIRV-Tools with `SPV_ENV_VULKAN_1_2`. Validation runs
before and after optimization. Every optimizing preset conditionally ends
with the grammar-aware trim-capabilities pass when the input module satisfies
the locally audited capability contract. Optimizer output is validated before
it is committed; optimizer failure or invalid output retains the already
validated input binary. Final feature reconciliation always uses the binary
that will actually be persisted.

`LUISA_SPIRV_OPT_LEVEL` selects the optimizer preset:
- Level 0 (`none`): disables optimization.
- Level 1 (`lightweight`): minimal cleanup (`adce`, `block-merge`, `simplification`, `dead-branch-elim`).
- Level 2 (default, `compute`): runs `register_compute_passes()`, which mirrors value/memory optimization without XIR-redundant or high-blowup whole-module transforms (`inline-exhaustive`, `merge-return`, `wrap-op-kill`, `eliminate-dead-functions`, `loop-unswitch`, or automatic `CreateLoopUnrollPass(false, 0)` on `LoopControl None`). `CreateLoopUnrollPass(true)` is registered only when the input module carries an explicit `Unroll` `OpLoopMerge` bit; `ScalarReplacement` is bounded by `LUISA_SPIRV_OPT_SROA_LIMIT` (default 100); value passes (`strength-reduction`, `boolean-simplify`, fast-math `float-div-opt`, cleanup) iterate up to `LUISA_SPIRV_OPT_MAX_ITERATIONS` (default 5), followed once post-convergence by FMA contraction under fast math.
- Level 3+ (`full`): runs stock `RegisterPerformancePasses(false)` plus a safe value-level tail (omitting default `loop-unswitch` and automatic unroll), plus post-convergence FMA contraction under fast math. `--loop-unswitch` remains opt-in via `LUISA_SPIRV_OPT_PASS_FLAGS`.
Use level 0 to isolate emission, never as a fix.

Useful diagnostics:

```bash
LUISA_DUMP_SOURCE=1             # dump XIR stages and Vulkan SPIR-V disassembly
LUISA_DUMP_SPV=1                # dump pre-optimization binary
LUISA_SPIRV_OPT_LEVEL=0         # isolate native emission
LUISA_SPIRV_LOOP_HINTS=0 # keep every OpLoopMerge at MaskNone (A/B; hashed into the shader cache key)
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 # reject any user-shader HLSL fallback
```

Use `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` for runtime or graphics
validation that specifically claims to exercise XIR -> SPIR-V. The guard does
not reject internal Vulkan builtins, which deliberately remain HLSL-generated.
It is parsed by every Vulkan build: a build without native XIR codegen fails
before compiling a user `Function`, a native build reports every unsupported
feature that requested the compatibility path, and AOT loading under the guard
accepts only an `XIR_SPIRV` artifact. Add `LUISA_DUMP_SOURCE=1` when the test
must prove fresh JIT code generation rather than execution of a valid cached
native artifact; source dumping forces the native compile path before cache
deserialization.

## Integer bit operations

XIR `CLZ`, `CTZ`, `POPCOUNT`, and `REVERSE` accept only `uint32` scalars or
vectors. Keep their emitter paths exact: `CLZ` uses `FindUMsb`, `CTZ` uses
`FindILsb`, and popcount/reverse lower directly to their core SPIR-V
instructions. Signed selection and narrow-integer promotion in these paths are
dead repair logic, not forward-compatible support. Runtime checks for
popcount/reverse must compare integer results directly; converting full-width
bit patterns to `float` loses enough precision to hide incorrect results.

## Floating-point contraction

When `ShaderOption::enable_fast_math` is false, every emitted floating
multiply/add/subtract that represents a source arithmetic operation must carry
`NoContraction`. This includes the component instructions used to expand
matrix arithmetic, the instructions used to expand floating reductions, and
native matrix multiply and outer-product instructions, not only scalar/vector
`OpFMul`, `OpFAdd`, and `OpFSub`. Decorating only a final
`OpCompositeConstruct` is invalid and does not protect its component
operations.

Precise `DOT` and the dot product inside vector `LENGTH`, `LENGTH_SQUARED`,
and `NORMALIZE` use explicit scalar products followed by ordered additions.
Each operation uses the input element type (including half and double) and
has `NoContraction`. A decorated `OpDot` still permits more accurate internal
evaluation and does not require each product to round separately. Fast math
continues to emit native `OpDot`. Keep rounding-sensitive runtime checks and
exact instruction/decoration counts; do not relax the precise numeric oracle
to accept the fused result.

When fast math contracts a multiply/add pair, select the multiply exactly once
and share that plan with instruction scheduling and FMA emission. An add may
have two multiply operands, and its first multiply may be multi-use while its
second is single-use. Deferring every single-use multiply whose user is an add
can therefore suppress a value that the chosen FMA still needs as its addend,
causing a use-before-definition failure. Only the selected single-use multiply
may be deferred; all other products must be emitted normally.

Emission policy and optimizer policy are separate and stay aligned.
`NoContraction` is the emitter's per-instruction veto, while the optimizer's
decision to run its non-IEEE-exact stages comes from
`ShaderOption::enable_fast_math` (`SpirvOptimizerOptions::enable_fast_math`) and
the contraction output form from the negotiated `shaderFmaFloat32` feature
(`use_khr_fma`), never from the module's decorations. With fast math the
`compute`/`full` presets run one extra post-convergence stage after the fixed
point — `FmaContractionPass` -> `EliminateDeadConstant` -> `CompactIds` — and
`FmaContractionPass` itself still honors `NoContraction` and still propagates
`RelaxedPrecision`; the output is `OpFmaKHR` when `shaderFmaFloat32` is enabled,
otherwise GLSL.std.450 `Fma`. A fast-math run that contracts nothing is still a
successful no-op: `SpirvOptimizerReport::fma_contraction_attempted` is true
while `fma_contraction_applied` stays false unless the post stage's bytes
differ from the pre-stage binary, so tests must assert both fields separately
instead of treating a successful stage as proof of contraction.

XIR generalizes `OUTER_PRODUCT` to two matrix operands for autodiff. That form
means `A * transpose(B)` and should lower to one `OpTranspose` followed by one
`OpMatrixTimesMatrix`; the ordinary vector form remains `OpOuterProduct`.

XIR `ROUND` has the C/C++ `round` contract: halfway cases round away from
zero and signed zero is preserved. Do not implement it as
`trunc(x + sign(x) * 0.5)`: the addition can round the float immediately below
`0.5` to `1.0`. Classify the fractional magnitude against an exactly typed
`0.5`, choose the adjacent integral magnitude, then copy the original sign
bit.

`SpirvRoundPolicy` (in `target_features.h`) selects between that portable
sequence and a single `GLSLstd450Round`. **The portable sequence is the
default**, because GLSL.std.450 leaves `Round`'s tie direction to the
implementation and at least one shipping driver (NVIDIA) rounds halves to even:
with the native form, `round(-2.5)` becomes `-2`, so the emitted module is
valid SPIR-V with *different semantics*, not a codegen bug. Only a device
qualified on the tie contract may select `NATIVE_GLSL_ROUND`; there is no
vendor for which that has been proven. The policy is a codegen preference, not
a target feature, so it must be part of the shader cache key (it is hashed by
`compute_shader_cache_md5` next to `buffer_float32_atomic_rmw_policy`).

The `ROUND` runtime oracle for `nextafter(±0.5)`, tie direction, and signed zero
is `vk_user_compute_vector_rounds_half_away_from_zero` in
`src/tests/unit/runtime/test_vk_spirv_codegen_path.cpp`. `test_device_math`
alone is not sufficient: its `round(def(2.5f))` inputs are compile-time
constants, so constant folding evaluates them and never exercises the emitted
instruction.

Run `test_vk_native_route_guard vk` in both native-XIR and LLVM/HLSL-only
Vulkan build trees. The native configuration verifies an explicit fallback
reason is rejected; the non-native configuration verifies the build-unavailable
diagnostic. The test isolates fatal diagnostics in a child process.

`test_vk_spirv_codegen_path` is a mixed suite, and the typed bindless cases are
native coverage. The typed `BUFFER_ONLY`, `TEXTURE2D_ONLY` and volume cases
install the strict guard themselves through a scoped
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and assert on the native
XIR-to-SPIR-V route, so they belong in strict runs, not in an exclusion list.
Only two cases deliberately take the compatibility route, both because an HLSL
`native_include` block is part of the contract they pin:
`vk_hlsl_writer_to_native_consumer_preserves_indirect_abi` and
`vk_indirect_host_empty_plan_is_a_true_noop`. A third declared case,
`vk_dxc_fallback_retains_typed_bindless_slot_abis`, is compiled only when
`LUISA_TEST_VK_HAS_DXC_COMPATIBILITY` is defined; the xmake test build leaves it
undefined, so that case is absent rather than excluded.
Run the whole suite once under Vulkan validation without the strict guard, then
run every remaining case individually with
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`. Do not clear the guard inside a
nominally strict case or describe an HLSL-shader case as native XIR output.
Keep the strict runner's explicit exclusion list synchronized with these case
names instead of documenting a case count that changes whenever coverage grows.

The Vulkan backend's shared `backend_print_code_enabled()` contract reads
`LUISA_DUMP_SOURCE`; it does not read the obsolete
`LUISA_COMPUTE_PRINT_CODE` variable.

## Floating-point remainder

Exact software remainder lowering on `x % y` (`BINARY_MOD`) is gated on strict
math (`!enable_fast_math`), matching HLSL codegen (`_lc_fmod` vs `fmod`). Under
`enable_fast_math = true`, `BINARY_MOD` lowers to native `OpFRem` with zero
`Int64` or loop overhead. `test_spirv_float_remainder` pins both strict and
fast-math emission contracts without a device, while `test_device_math` and
`test_hip_fmod` verify runtime execution.

- Under strict math (`enable_fast_math = false`), binary16 widens to binary32,
  reuses the exact binary32 reduction, and narrows once (exact since the exact
  remainder of two binary16 values fits in binary16).
- Under strict math, binary32 uses a pure 32-bit structured `OpLoopMerge`
  shift-and-reduce loop processing up to 8 exponent bits per iteration: since
  `rem < safe_y_significand <= 2^24`, `rem << step < 2^32`, eliminating
  `OpCapability Int64` and `OpFDiv` completely.
- Under strict math, binary64 uses a bounded 41-iteration structured
  `OpLoopMerge` chunked reduction: for
  `s = clamp(exponent(|x|) - exponent(|y|) - 52, 0, 2045)`, `t = |y| * 2^s`, the
  floored quotient `floor(r / t)` fits below `2^53`, and `r - q * t` is formed
  with unconditional `NoContraction` on `Fma`, `OpFMul`, and `OpFAdd`, selecting
  between `q` and `q + 1` residuals; the structured loop stops early once all
  invocations are finished or bounded at 41 chunks.

## Testing expectations

Planner-only tests are appropriate for arithmetic limits and malformed input,
but accepted codegen features also need an exact `compile_spirv_xir` or AST
compile test so SPIRV-Tools validates the emitted module.

Important targets under `src/tests/unit/ext/`:

- `test_spirv_xir_dialect`
- `test_spirv_pointer_legalization`
- `test_spirv_control_flow_plan`
- `test_spirv_instruction_layout`
- `test_spirv_buffer_layout`
- `test_spirv_aggregate_indices`
- `test_spirv_target_feature_codegen`
- `test_spirv_runtime_target_plan`
- `test_spirv_raw_float_constants`
- `test_spirv_float_remainder`
- `test_spirv_optimizer`
- `test_vk_ext_shader_caps` and `test_vk_extension_registry` (the shader-capability tables)
- `test_vk_android_plan`
- `test_vk_saved_argument_contract`
- `test_vk_shader_binary_contract`
- `test_argument_block_layout`

For an accepted feature, test both the exact boundary and one-over rejection.
For control flow, include the physical edge shape that caused the bug rather
than only calling a mirror helper. For persisted/runtime ABI changes, test both
writer and reader/consumer order where feasible.

## Change checklist

Before handing off a native SPIR-V change:

1. Keep generic XIR passes target-neutral.
2. Establish a checked, immutable plan before target mutation.
3. Validate numeric width, signedness, alignment, count, and storage class at
   every subsystem boundary.
4. Update dialect classification and diagnostics.
5. Add planner rejection tests and validator-backed accepted tests.
6. Validate pre- and post-optimization SPIR-V.
7. Run Vulkan validation layers for runtime layout/descriptor changes.
8. Exercise the Vulkan XIR -> SPIR-V path, not an LLVM/HLSL fallback.
9. Keep cached artifact and logical-device feature contracts in sync.
10. Update this skill when an architectural invariant changes.
