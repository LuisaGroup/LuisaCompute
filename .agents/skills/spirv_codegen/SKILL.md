---
name: spirv_codegen
description: SPIR-V backend codegen: XIR to SPIR-V with glslang Builder, bindings, and ray query.
---

# SPIR-V Codegen

Translates XIR→SPIR-V using glslang's `spv::Builder`. Located in `src/backends/common/spirv/spirv_codegen/`.

## Directory Layout

```
src/backends/common/spirv/spirv_codegen/
├── lc_spirv_pch.h       # PCH: all luisa/xir headers + SPIRV/SpvBuilder.h + SPIRV/disassemble.h
├── property.h           # Re-exports hlsl::ShaderVariableType / hlsl::Property
├── entry.h              # SpirvCodegenEntry class, SpirvResult struct, method declarations
├── entry.cpp            # compile_spirv() static entry point
├── emit.cpp             # Usage analysis, block/function/kernel/callable emission, value emission
├── type.cpp             # _convert_type() + _convert_laid_out_type()
├── instruction.cpp      # Per-instruction dispatch + arithmetic, atomic, resource, thread_group, ray query
├── condition_inst.cpp   # Control flow: if, loop, simple_loop, switch, branch, conditional_branch
├── bind.cpp             # Descriptor bindings, global variables, decorations
└── utils.h/cpp          # AST→XIR translation + optimization pipeline
```

Build: xmake target `lc-spirv` (static lib `luisa-spirv.lib`), deps `lc-vstl`, `lc-runtime`, `lc-glslang`, `SPIRV-Tools-opt`.

## Architecture

### Three-Phase Pipeline

1. **Analysis** (`_analyze_instruction_usage`): post-order traversal collecting `used_types`, `used_constants`, `used_functions_post_order`, print info.
2. **Emission** (`emit()`): convert types → emit constants → emit functions (kernel+callables) → `postProcess(false)` → `dump()` + disassembly.
3. **Upstream AST→XIR** (`luisa_spirv_backend_translate_ast_to_xir` in `utils.cpp`): AST→XIR translation followed by optimization pipeline (Phases A/B/C).

### Key Class

```cpp
namespace lc::spirv {  // using namespace luisa; using namespace luisa::compute;

class SpirvCodegenEntry {
    StringScratch &_scratch;
    std::unique_ptr<spv::Builder> _builder_ptr;  // intentionally leaked via .release()
    spv::SpvBuildLogger _logger;
    spv::Builder &_builder;  // reference to *_builder_ptr

    // Core maps
    unordered_map<const Type*, spv::Id> _type_map;
    unordered_map<const xir::Value*, spv::Id> _value_map;
    unordered_map<const xir::Function*, spv::Function*> _function_map;
    unordered_map<const xir::BasicBlock*, spv::Block*> _block_map;
    // _loop_header_info maps loop prepare/dispatch block to (header, continue_target)
    unordered_map<const xir::BasicBlock*, pair<spv::Block*, spv::Block*>> _loop_header_info;
    // _loop_header_redirect: redirects branches to body/prepare/dispatch to the correct SPIR-V block (used by all loop types and ray_query_loop)
    unordered_map<const xir::BasicBlock*, spv::Block*> _loop_header_redirect;
    unordered_set<const xir::BasicBlock*> _emitted_blocks;
    unordered_set<spv::Id> _used_merge_blocks;  // prevent merge block reuse

    // Print support
    unordered_map<const xir::PrintInst*, PrintInfo> _print_info;
    PrintFormatVector _print_formats;  // vector<pair<string, const Type*>>
    bool _requires_printing{false};

    // Bindings
    SpirvResult::Properties _properties;
    vector<spv::Id> _property_ids;
    bool _use_tex2d_bindless, _use_tex3d_bindless, _use_buffer_bindless;

    // Heap/builtin handles
    spv::Id _buffer_heap_id{NoResult}, _tex2d_heap_id{NoResult}, _tex3d_heap_id{NoResult};
    spv::Id _glsl450{NoResult};
    spv::Instruction *_entry_point_inst{nullptr};
    spv::Id _global_invocation_id_var{NoResult};
    unordered_map<spv::BuiltIn, spv::Id> _builtin_var_map;

    // Resource metadata
    unordered_map<spv::Id, bool> _is_storage_image_map;  // image var → isStorageImage
    unordered_map<spv::Id, spv::Id> _accel_instance_buffer_map;  // accel var → instance buffer
    unordered_map<spv::Id, spv::Id> _rq_proceed_result;  // rq SSA id → last proceed bool
    unordered_map<const xir::Function*, vector<bool>> _callable_arg_used;
    unordered_set<const Type*> _needs_atomic_buffer_types;
    unordered_map<const Type*, spv::Id> _laid_out_type_map;  // for buffer struct layouts
    xir::UniformityAnalysis _uniformity;
};
}
```

### Entry Point

```cpp
// entry.cpp: compile_spirv() — static method, the only public API
SpirvResult SpirvCodegenEntry::compile_spirv(Function kernel, const ShaderOption &opt) {
    auto xir_module = luisa_spirv_backend_translate_ast_to_xir(kernel, opt);
    StringScratch scratch;
    SpirvCodegenEntry codegen{scratch, true};
    codegen.generate_binding(kernel);       // Step 1: descriptor layout
    codegen.emit(xir_module.get(), ...);    // Step 2: SPIR-V emission
    codegen._builder.dump(words);           // Step 3: binary
    codegen._builder_ptr.release();         // Intentional leak to avoid dtor crash
    return SpirvResult{words, props, printers, useTex2DBindless, useTex3DBindless, useBufferBindless};
}
```

## Type Conversion (`type.cpp`)

Two conversion paths:

### `_convert_type(type, usage)` — Normal types

| Type Tag | SPIR-V Call |
|---|---|
| BOOL/INT*/UINT*/FLOAT* | `makeBoolType()` / `makeIntType(w)` / `makeUintType(w)` / `makeFloatType(w)` |
| VECTOR | `makeVectorType(elem, dim)` |
| MATRIX | `makeMatrixType(elem, dim, dim)` |
| ARRAY | `makeArrayType(elem, size_id, 0)` |
| STRUCTURE | `makeStructType(member_types, {}, "Struct", false)` |
| BUFFER | `makeRuntimeArray(elem_type)` wrapped in `makeStructType`, Block+Offset+ArrayStride decorations. Typed unless atomic. Matrix→ColMajor+MatrixStride. Falls back to `makeUintType(32)` for atomic-needed buffers. |
| TEXTURE | `makeImageType(sampled_type, dim, false, false, false, sampled, fmt)`. `sampled=2` (storage) if writable, `sampled=1` if read-only. Format: Rgba32f/Rgba32i/Rgba32ui. |
| BINDLESS_ARRAY | `makeRuntimeArray(uint32)` in struct{Block, ArrayStride=4, Offset=0, NonWritable} |
| ACCEL | `makeAccelerationStructureType()`, adds `SPV_KHR_ray_query` ext + `RayQueryKHR` cap |
| CUSTOM | `makeRayQueryType()` for `LC_RayQueryAll`/`LC_RayQueryAny`, adds `SPV_KHR_ray_query` ext |

### `_convert_laid_out_type(type)` — Layout-decorated types for buffer structs

Recursively adds `ArrayStride`, `Offset`, `ColMajor`, `MatrixStride` decorations. Used for structured/array buffer element types when those elements are structs or arrays.

## Value Emission (`emit.cpp`)

```cpp
spv::Id _emit_constant(const xir::Constant *c);   // scalar→make*Constant, vector/matrix/array/struct→makeCompositeConstant
spv::Id _emit_value(const xir::Value *value);      // dispatches by DerivedValueTag
```

| Tag | Handling |
|---|---|
| CONSTANT | `_emit_constant()` |
| UNDEFINED | `createUndefined(type)`. **Not cached** in `_value_map` (must dominate all uses). |
| SPECIAL_REGISTER | Input var + BuiltIn decoration + `createLoad`. Cached in `_builtin_var_map`. **Not cached** in `_value_map` (dominance). Special case: `DISPATCH_SIZE` reads from push constant `_property_ids[0]`. |
| ARGUMENT (resource) | `_resolve_resource_argument()` → `_property_ids` lookup |
| ARGUMENT (value) | Function param ID from kernel/callable emission |
| FUNCTION/BLOCK/INSTRUCTION | Pre-mapped in `_value_map`/`_block_map`/`_function_map` |

Special registers: `THREAD_ID`→LocalInvocationId, `BLOCK_ID`→WorkgroupId, `DISPATCH_ID`→GlobalInvocationId, `BLOCK_SIZE`→WorkgroupSize, `WARP_SIZE`→SubgroupSize, `WARP_LANE_ID`→SubgroupLocalInvocationId.

### Access Chain Helper

```cpp
spv::Id _create_access_chain(spv::StorageClass storage, spv::Id base,
                             const std::vector<spv::Id> &indices, bool nonuniform = false);
```
Saves/restores `_builder.getAccessChain()`. If `nonuniform`, adds `NonUniformEXT` decoration to indices and result.

### Uniformity Helper

```cpp
spv::Id _ensure_type(spv::Id value, spv::Id target_type);
```
Ensures value matches target_type. Handles: same-class float (FConvert), int (SConvert/UConvert/Bitcast), cross-signedness int, float↔int (ConvertFToS/U, ConvertSToF/UToF), bool↔int (OpSelect or INotEqual). Falls back to OpBitcast.

## Instruction Emission (`instruction.cpp`)

### `_emit_instruction()` — Main dispatch

```cpp
ALLOCA → createVariable(Function/Workgroup). Workgroup alloca added to entry point.
LOAD   → createLoad (RayQueryKHR: pass-through; spec forbids OpLoad on OpTypeRayQueryKHR).
STORE  → createStore (RayQueryKHR: remap `_value_map[variable] = val` so subsequent loads resolve to the source variable; OpCopyMemory is forbidden on OpTypeRayQueryKHR since Rev 15). Handles type mismatch (scalar→vector smear, bitcast).
GEP    → _create_access_chain with builder storage class.
ARITHMETIC → _emit_arithmetic_inst
CALL    → createFunctionCall; skips unused resource args; creates temp vars for reference args (non-alloca/non-param).
CAST    → OpBitcast/OpSelect/OpConvert*/OpFConvert/OpSConvert/OpUConvert. Bool↔non-bool: via uint intermediate.
IF/LOOP/SIMPLE_LOOP/SWITCH/BRANCH/CONDITIONAL_BRANCH → condition_inst.cpp
RETURN  → makeReturn (with optional value)
BREAK   → createBranch to target (inline, uses _get_or_create_block)
CONTINUE → createBranch to target (inline)
UNREACHABLE → OpUnreachable
ATOMIC  → _emit_atomic_inst
RESOURCE_QUERY → _emit_resource_query_inst
RESOURCE_READ  → _emit_resource_read_inst
RESOURCE_WRITE → _emit_resource_write_inst
THREAD_GROUP   → _emit_thread_group_inst
PHI     → ERROR (must be eliminated)
RAY_QUERY_LOOP     → _emit_ray_query_loop_inst
RAY_QUERY_DISPATCH → _emit_ray_query_dispatch_inst
RAY_QUERY_OBJECT_READ  → _emit_ray_query_object_read_inst
RAY_QUERY_OBJECT_WRITE → _emit_ray_query_object_write_inst
PRINT   → no-op (not supported in SPIR-V)
AUTODIFF_* → ERROR (must be eliminated)
CLOCK/ASSERT/ASSUME/DEBUG_BREAK/OUTLINE/RASTER_DISCARD/RAY_QUERY_PIPELINE → NOT_IMPLEMENTED
```

### Arithmetic (`_emit_arithmetic_inst`)

Helper lambdas: `unary(op)`, `binary(op)`, `glsl(builtin,...)` (createBuiltinCall), `glsl_typed(f,s,u,...)`.

**Constant folding**: `BINARY_ADD` (OpIAdd via SpecConstantOp), `BINARY_MUL` (OpIMul via SpecConstantOp), `BINARY_EQUAL`/`BINARY_NOT_EQUAL` (OpIEqual/OpINotEqual via SpecConstantOp).

**Strength reduction**: `BINARY_MUL(x, 2^n)` → `OpShiftLeftLogical`, `BINARY_DIV(x, 2^n)` → `OpShiftRightArithmetic`.

**SELECT**: SPIR-V order (cond, true, false). Constant-fold if condition is constant bool.

**AGGREGATE peephole**: Detects Extract-from-same-vector pattern and emits `OpVectorShuffle` via `createRvalueSwizzle`.

**SHUFFLE peephole**: If all indices are constants, emits `createRvalueSwizzle`; otherwise `createVectorExtractDynamic` per element.

**EXTRACT**: All-constant → `createCompositeExtract`; vector with dynamic → `createVectorExtractDynamic`; array/matrix with dynamic → temp var + access chain + load.

**INSERT**: Constant indices → `createCompositeInsert`.

**Reductions** (SUM/PRODUCT/MIN/MAX): Manual extract + per-element combine loop.

**Matrix ops**: `COMP_NEG`→row-wise FNegate, `COMP_ADD/SUB/MUL/DIV`→row-wise ops (scalar: smear first), `LINALG_MUL`→VectorTimesMatrix/MatrixTimesVector/MatrixTimesMatrix, `DETERMINANT`→GLSLstd450Determinant, `TRANSPOSE`→OpTranspose, `INVERSE`→GLSLstd450MatrixInverse.

**Boolean ops**: `AND`→OpLogicalAnd (if bool), `OR`→OpLogicalOr, `XOR`→OpLogicalNotEqual, `NOT`→OpLogicalNot.

**EXP10** = Exp2(x * log2(10)). **LOG10** = Log2(x) * inv_log2_10. **POW_INT** = Pow(x, ConvertSToF(y)). **ROUND** = Trunc(x + copysign(0.5, x)). **ROTATE_LEFT/RIGHT** = shift_1 | shift_2.

### Atomic (`_emit_atomic_inst`)

Scope=Device, Semantics=MaskNone. Base pointer + optional `createAccessChain`.

**Non-scalar buffer handling**: When `base->type()->is_buffer()` with a non-uint32 element type, computes word offsets from indices, accessing the underlying uint32 array.

**Float atomics**: Uses SPIR-V extensions:
- float16 add: `SPV_EXT_shader_atomic_float16_add`, `OpAtomicFAddEXT`
- float32 add: `SPV_EXT_shader_atomic_float_add`, `OpAtomicFAddEXT`
- float64 add: `SPV_EXT_shader_atomic_float_add`, `OpAtomicFAddEXT`
- float min/max: `SPV_EXT_shader_atomic_float_min_max`, `OpAtomicFMinEXT`/`OpAtomicFMaxEXT`

**Float atomic fallback** (non-scalar buffers): CAS loop with `OpAtomicLoad` + `OpAtomicCompareExchange` on uint32 values via bitcast.

**Float compare-exchange on scalar buffers**: CAS loop with `OpAtomicLoad` + `OpAtomicExchange` (avoids pointer bitcast that crashes NVIDIA drivers).

### Resource Read/Write

**Typed buffer** (has non-null element type): Direct access via `createAccessChain(0, index)` + `createLoad`/`createStore`. Uses `OpCopyLogical` for type mismatch.

**Byte buffer / bindless**: Word-level access. Computes `word_offset = index * word_count`. Sub-word types (bool, int8, etc.): read-modify-write with shift+mask.

**Buffer read (`_emit_buffer_read_impl`)**: Recursively handles vectors, matrices, structures, arrays. Sub-word struct members: read containing word, shift, mask, bitcast.

**Buffer write (`_emit_buffer_write_impl`)**: Recursively handles same types. Sub-word struct members: read-modify-write with bitfield insertion.

**Texture read**: `OpImageRead` (storage, +`StorageImageReadWithoutFormat` capability) or `OpImageFetch` (sampled).

**Texture write**: `OpImageWrite` (+`StorageImageWriteWithoutFormat` capability).

Note: `StorageImageReadWithoutFormat` and `StorageImageWriteWithoutFormat` are core capabilities in SPIR-V 1.5 (no extension needed). `ImageQuery` capability is required for `OpImageQuerySize`/`OpImageQuerySizeLod`.

**Bindless buffer**: Resolves from bindless_array (3 uint32s per slot: buffer_index, tex2d_index, tex3d_index), then accesses buffer_heap.

**Bindless texture sample**: Resolves texture index from bindless_array, selects sampler from sampler_heap (either packed in upper 4 bits or from filter/address params), creates `OpSampledImage`, calls `createTextureCall` with `noImplicitLod=true`.

### Thread Group

- `SYNCHRONIZE_BLOCK`: `createControlBarrier(Workgroup, Workgroup, AcquireRelease|WorkgroupMemory)`
- `WARP_IS_FIRST_ACTIVE_LANE`: `OpGroupNonUniformElect` (+`GroupNonUniform`)
- `WARP_ACTIVE_ALL/ANY`: `OpGroupNonUniformAll/Any`
- `WARP_ACTIVE_BIT_AND/OR/XOR`: `OpGroupNonUniformBitwiseAnd/Or/Xor` with `GroupOperation::Reduce`
- `WARP_ACTIVE_SUM/PRODUCT`: `OpGroupNonUniformFAdd/FMul` or `IAdd/IMul` (+`GroupNonUniformArithmetic`)
- `WARP_ACTIVE_MIN/MAX`: `OpGroupNonUniformFMin/FMax` or `SMin/SMax/UMin/UMax`
- `WARP_ACTIVE_ALL_EQUAL`: `OpGroupNonUniformAllEqual` (+`GroupNonUniformVote`)
- `WARP_ACTIVE_COUNT_BITS`: `OpGroupNonUniformBallot` + `OpGroupNonUniformBallotBitCount` (+`GroupNonUniformBallot`)
- `WARP_ACTIVE_BIT_MASK`: `OpGroupNonUniformBallot` (returns uvec4)
- `SHADER_EXECUTION_REORDER`: ignored (no-op)

### Ray Query

**`RAY_TRACING_QUERY_ALL/ANY`**: Creates ray query variable (Function storage), `OpRayQueryInitializeKHR(rq_var, accel, ray_flags, cull_mask, origin, t_min, dir, t_max)`, returns rq_var. Adds `SPV_KHR_ray_query` ext + `RayQueryKHR` cap. Per spec, acceleration structures must not be used with `OpPhi`/`OpSelect`.

**`RAY_TRACING_TRACE_CLOSEST/ANY`**: Initialize, `OpRayQueryProceedKHR`, `OpRayQueryGetIntersectionTypeKHR`. Closest-hit: branch on triangle vs non-triangle, build Hit result. Any-hit: return `committed_type != 0`.

**`RAY_TRACING_INSTANCE_TRANSFORM`**: Reads 3 float4 rows from accel instance buffer at `instance_index * 16` words, constructs 4x3 matrix.

**`RAY_TRACING_SET_INSTANCE_*`**: Writes transform (3 float4 rows), visibility_mask (word 13), user_id (word 12), opacity (word 14, encoded 4=opaque 8=non-opaque) to instance buffer.

**Ray Query Loop**: `_loop_header_info[dispatch_block] = (header, continue_block)`, `_loop_header_redirect[dispatch_block] = continue_block`. Creates loop header + continue block with `OpLoopMerge`. Per SPIR-V structured-CFG rules, loop headers must be the target of a back-edge branch; the continue_block→header branch satisfies this.

**Ray Query Dispatch**: `OpRayQueryProceedKHR` → branches to check_block → `OpRayQueryGetIntersectionTypeKHR` on candidate → branches surface/procedural. Temporarily redirects `_loop_header_redirect` to dispatch_merge_block, restoring after.

**Ray Query Object Read**: Uses `OpRayQueryGetWorldRayOriginKHR/DirectionKHR`, `OpRayQueryGetRayTMinKHR`, `OpRayQueryGetIntersectionTKHR/InstanceIdKHR/PrimitiveIndexKHR/BarycentricsKHR/TypeKHR` with committed/candidate constant.

**Ray Query Object Write**:
- `COMMIT_TRIANGLE` → `OpRayQueryConfirmIntersectionKHR`
- `COMMIT_PROCEDURAL` → `OpRayQueryGenerateIntersectionKHR(rq_obj, dist)`
- `TERMINATE` → `OpRayQueryTerminateKHR`
- `PROCEED` → `OpRayQueryProceedKHR`, stores result in `_rq_proceed_result` map</parameter>

## Control Flow (`condition_inst.cpp`)

- **If** (`_emit_if_inst`): `bind_or_get` blocks, synthetic merge block if merge already used, `OpSelectionMerge`, `OpBranchConditional`, emit true/false, branch to merge.
- **Loop** (`_emit_loop_inst`): creates synthetic header block, sets `_loop_header_redirect[prepare]=header`, branches header→prepare, `OpLoopMerge(merge, update)` in header.
- **Simple Loop** (`_emit_simple_loop_inst`): do-while. Creates header + continue_block, `_loop_header_redirect[body]=continue_block`, `OpLoopMerge(merge, continue_block)` in header, body→continue→header.
- **Switch** (`_emit_switch_inst`): `bind_or_get` case blocks + default + merge, synthetic merge if needed, `OpSelectionMerge`, `OpSwitch` with case values + default, emit each case, branch to merge.
- **Branch** (`_emit_branch_inst`): resolves targets through `_resolve_branch_target`, including loop header redirects and trivial XIR forwarders that lead to an enclosing loop merge/continue.
- **CondBranch** (`_emit_conditional_branch_inst`): uses the same target resolver for both targets.

Key patterns:
- `_used_merge_blocks`: prevents reusing merge block IDs (SPIR-V requires unique merge targets).
- `_loop_header_redirect`: For all loop types, redirects branches to prepare/body/dispatch to the correct SPIR-V block (header or continue_block).
- `_loop_boundary_stack`: lets branch emission collapse an empty XIR block that only forwards to an enclosing loop merge/continue. Do not collapse the active selection merge target during normal branch resolution; normal `OpSelectionMerge` blocks must remain materialized.
- `_branch_target_redirect`: scoped override for one-sided `IfInst` arm emission, used when a real XIR arm terminator branches to an empty merge block that forwards to a loop boundary.
- `_forwarded_blocks`: XIR blocks whose branches have been collapsed away; do not later emit them as pending top-level blocks after the loop-boundary context is gone.
- `_emit_block()` returns `true` only when it emitted that block body in this call. `false` also covers blocks that are already in progress, not just blocks that are fully emitted; callers such as `IfInst` arm emission must not add synthetic terminators after a `false` return.
- `_emit_if_inst()` postdom rule: if both arms can reach the XIR merge and either arm says the merge post-dominates, keep the real merge as the SPIR-V selection merge. Otherwise, unreachable subpaths can make a real two-sided merge look one-sided and produce branch-into-selection errors.
- Terminal arms (`return`, `unreachable`, `raster_discard`) are not fall-through continuations. Keep the real merge for the live arm instead of using the terminal arm block as the selection merge.
- `_block_map` binding: blocks are allocated lazily via `_get_or_create_block()` or `bind_or_get()`.

## Binding Generation (`bind.cpp`)

`generate_binding(Function kernel)`:

1. **Detect cbuffer_non_empty**: any arg not a resource/builtin → cbuffer has data.
2. **Detect bindless usage**: checks `kernel.propagated_builtin_callables()` against CallOp sets for buffer/tex2d/tex3d bindless.
3. **Register indexer**: `RegType` enum (`CBV=0, UAV=1, SRV=2`), flat counter `reg_count` starting after fixed-position items.

### Fixed-position properties

| Index | Type | Details |
|---|---|---|
| 0 | ConstantValue (PushConstant) | `space=0 reg=0 size=1`, uint4 type |
| 1 | SamplerHeap | `space=1 reg=0 size=16`, sampler array |
| 2 | StructuredBuffer (opt) | CBuffer for non-resource args, `space=0 reg=0 size=1` |

### Bindless heaps (space 2+)

| Condition | Type | Name |
|---|---|---|
| `_use_buffer_bindless` | SRVBufferHeap | `bdls` |
| `_use_tex2d_bindless` | SRVTextureHeap | `tex2d_heap` |
| `_use_tex3d_bindless` | SRVTextureHeap | `tex3d_heap` |

### Per-argument bindings

| Arg Tag | Read-only | Writable |
|---|---|---|
| TEXTURE | SRVTextureHeap, space=0 | UAVTextureHeap, space=0 |
| BUFFER | StructuredBuffer, space=0 | RWStructuredBuffer, space=0 |
| BINDLESS_ARRAY | StructuredBuffer, space=0 | — |
| ACCEL | SPIRVAccel + StructuredBuffer (instances) | RWStructuredBuffer |
| CUSTOM (IndirectDispatch) | — | RWStructuredBuffer |

### SPIR-V global variable creation

- **ConstantValue**: PushConstant storage, struct{uint4}, Block+Offset decoration.
- **SamplerHeap**: UniformConstant storage, array of sampler type, DescriptorSet+Binding.
- **StructuredBuffer/RWStructuredBuffer**: StorageBuffer storage. Typed: struct{runtime_array(elem_type)} with Block+ArrayStride+Offset+Coherent (if writable)+NonWritable (if read-only)+ColMajor+MatrixStride decorations. Bindless_array uses `_convert_type(Type::from("bindless_array"))`. Untyped: struct{runtime_array(uint32)}.
- **SRVTextureHeap**: UniformConstant storage, Image type with sampled=1, possibly RuntimeArray with `SPV_EXT_descriptor_indexing`. Stores `_tex2d_heap_id`/`_tex3d_heap_id`.
- **UAVTextureHeap**: Same but sampled=2 (storage image). `StorageImageArrayNonUniformIndexingEXT` capability.
- **SRVBufferHeap/UAVBufferHeap**: StorageBuffer storage, struct{runtime_array(uint32)} with Block+ArrayStride decorations. RuntimeArray with `SPV_EXT_descriptor_indexing`. Stores `_buffer_heap_id`.
- **SPIRVAccel**: UniformConstant storage, AccelerationStructure type.

### Print buffers
If `kernel.requires_printing()`: adds 2x RWStructuredBuffer (`_printCounter`, `_printBuffer`).

## Kernel Emission (`emit.cpp: _emit_kernel`)

1. Run `_uniformity.analyze(kernel)`.
2. Create function with void return, "main" name, no params (entry point can't have params).
3. **Load non-resource args from cbuffer**: Reads from `_property_ids[2]` (StructuredBuffer). Computes aligned offsets, handles sub-word types (bool: compare, int8/etc: truncate+bitcast), multi-word types via `_emit_buffer_read_impl`.
4. **Entry point**: `addEntryPoint(GLCompute, func, "main")`, adds all `_property_ids` as operands.
5. **Execution mode**: `LocalSize(block_size)`.
6. **Dispatch bounds check**: Creates early-return for threads with `GlobalInvocationId >= DispatchSize` (all 3 components). Uses `OpSelectionMerge` + `OpBranchConditional` → early return block vs body block. Re-maps XIR body block to the body block.
7. **Emit body block**.
8. **Terminate**: `makeReturn(false)` if not already terminated.

## Callable Emission (`emit.cpp: _emit_callable`)

1. Run `_uniformity.analyze(callable)`.
2. Build param types:
   - **Resource args**: pointer type (StorageBuffer for buffer/bindless_array, UniformConstant for accel/texture). Adds `VariablePointersStorageBuffer` capability.
   - **Reference args**: Function pointer type.
   - **Value args**: normal type.
3. Skip unused resource args (to avoid type mismatches with kernel globals).
4. Map function param IDs to arguments in `_value_map`.
5. Mark texture params as non-storage images in `_is_storage_image_map`.
6. Emit body block.

## AST→XIR Translation & Optimization Pipeline (`utils.cpp`)

`luisa_spirv_backend_translate_ast_to_xir(Function kernel, ShaderOption &option)`:

### Phase A (structured-CFG alloca-form)

1. `ast_to_xir_translate(kernel, {})` — produces structured alloca-form XIR
2. `dce` → `local_store_forward` → `local_load_elimination` → `dce`
3. `algebraic_simplify` → `const_fold` → `dce`
4. `promote_ref_arg` → `sroa` → `loop_unroll` → `dce`

### CFG Normalization (unless `LUISA_XIR_DISABLE_NORMALIZE_CFG=1`)

1. `lower_ray_query_loop_to_loop`
2. **`destructure_cfg`**: structured → unstructured (if→branch+cond_br, loop→branch+cond_br, break/continue→branch)
3. **Phase B** (SSA opts on unstructured): `mem2reg` → `algebraic_simplify` → `const_fold` → `dce` → `local_store_forward` → `local_load_elimination` → `dce`
4. `unused_callable_removal`
5. `simplify_cfg`

### CFG Restructuring (unless `LUISA_XIR_DISABLE_RESTRUCTURE_CFG=1`)

1. `reg2mem` (phi→alloca+load/store)
2. **`restructure_cfg`**: unstructured → structured
3. `dce` (post-restructure: removes orphan blocks left by restructure_cfg)
4. `reg2mem` (mid)
5. **Phase C**: `dce` → `local_store_forward` → `local_load_elimination` → `const_fold` → `dce`
6. `reg2mem` (post, final phi elimination before SPIR-V emission)

### Environment variables

| Variable | Effect |
|---|---|
| `LUISA_DUMP_SOURCE=1` | Dump XIR at each stage + final SPIR-V disasm |
| `LUISA_SPIRV_DUMP_OPT_STATS=1` | Log per-pass timing and counts |
| `LUISA_XIR_DISABLE_NORMALIZE_CFG=1` | Skip destructure/mem2reg/restructure (debug) |
| `LUISA_XIR_DISABLE_RESTRUCTURE_CFG=1` | Skip destructure+restructure, keep lower_ray_query_loop_to_loop (debug) |
| `LUISA_XIR_DISABLE_OPTIMIZATION=1` | Skip all XIR optimization passes (Phase A, post-inline, CFG normalization). Returns raw AST→XIR module directly. Use to isolate bugs: if bug disappears → XIR pass bug; if bug persists → SPIR-V codegen/backend bug. |

## SPIR-V Optimization (spv-opt)

After emission, `compile_spirv()` runs `spvtools::Optimizer` with `RegisterPerformancePasses()` (inline, DCE, scalar replacement, dead branch elimination, local access chain conversion, etc.). This catches SPIR-V-specific patterns that emerge after restructure_cfg and emission. Typical size reduction: 60-90%.

The optimizer links via `SPIRV-Tools-opt` (static lib from `src/ext/SPIRV-Tools`).

## Vendor-Specific Codegen

`compile_spirv(kernel, option, use_native_float_atomics)` accepts a vendor hint:

| Vendor | ID | `use_native_float_atomics` | Reason |
|---|---|---|---|
| NVIDIA | `0x10de` | `true` | Native `OpAtomicFAddEXT` is fast |
| AMD | `0x1002` | `false` | CAS loop is 5x faster than hardware float atomics on RDNA |
| Intel | `0x8086` | `false` | Conservative default |

The vk backend passes this based on `_vk_device->properties.vendorID`.

## Result & Debugging

```cpp
// entry.h
struct SpirvResult {
    vector<uint32_t> spv_bin;
    vector<Property> properties;                          // consumed by Vulkan descriptor allocator
    vector<pair<string, const Type*>> printers;           // host-side print formatting
    bool useTex2DBindless, useTex3DBindless, useBufferBindless;
};
```

### `LUISA_DUMP_SOURCE=1`

Set env var to dump codegen results. In `Device::create_shader()`:

- **XIR→SPIRV path** (`LUISA_XIR_TO_SPIRV`): logs binary size + property bindings. `print_code()`: writes `spv_code_<name>.spvasm` (XIR→SPIRV disassembly) + `spv_code_hlsl_<name>.spvasm` (HLSL→DXC→SPIRV for comparison). HLSL writes to `hlsl_output_<name>.hlsl`.
- **HLSL-only path**: `compile_only` + `print_code()`: writes `hlsl_output_<name>.hlsl`.

All files use `"wb"` (overwrite) mode — no append, no need to delete old files.

**Naming priority** (consistent across all backends):
1. `ShaderOption::name` — user-provided shader name
2. `Function::name()` — kernel/callable debug name
3. `Function::hash()` formatted as hex — fallback (e.g. `spv_code_a1b2c3d4e5f6a7b8.spvasm`)

For the VK `ComputeShader::compile()` path (no `Function` available), falls back to MD5 hash of the generated HLSL code when `file_name` is empty.

When `LUISA_XIR_TO_SPIRV` is undefined, backend falls back to HLSL codegen + DXC.

## Adding New Instructions

1. **Arithmetic op**: add case in `_emit_arithmetic_inst`, map to `spv::Op`/`GLSLstd450*`
2. **Resource op**: add to `_emit_resource_read/write/query_inst`
3. **Thread-group op**: add to `_emit_thread_group_inst`
4. **Control flow**: add method in `entry.h`, implement in `condition_inst.cpp`, dispatch from `_emit_instruction`
5. **Ray query op**: add to `_emit_ray_query_object_read/write_inst` or `_emit_ray_query_dispatch_inst`
6. **Binding support**: update `generate_binding()` in `bind.cpp`
7. **New type**: update `_convert_type()` in `type.cpp`

All paths must: use `_convert_type(inst->type())` for result, `_emit_value(op)` for operands, store in `_value_map` (or `spv::NoResult` for void).

## Debugging `_emit_value` Assertion: "should have been pre-mapped"

Location: `emit.cpp:323`, inside `_emit_value()`, the fallthrough case for `ARGUMENT` (non-resource, not in map) / `FUNCTION` / `BASIC_BLOCK` / `INSTRUCTION`.

### Root Cause

`_emit_value` is the central XIR→SPIR-V value resolver. It expects **structural/non-leaf values** to be pre-registered in `_value_map` *before* any instruction uses them as an operand. These are pre-mapped by:

| Tag | Pre-mapped by | When |
|---|---|---|
| `FUNCTION` (kernel) | `_emit_kernel()` line 402 | During kernel emission |
| `FUNCTION` (callable) | `_emit_callable()` line 617 | During callable emission |
| `ARGUMENT` (value/ref) | `_emit_kernel()` cbuffer load (lines 454–463) or `_emit_callable()` param mapping (line 623) | During function prologue |
| `ARGUMENT` (resource) | `_resolve_resource_argument()` → `_value_map.emplace` in `instruction.cpp:1178` | On first use |
| `INSTRUCTION` | `_emit_instruction()` via `set_result` lambda (line 2926–2928) | When instruction is emitted |
| `BASIC_BLOCK` | Should never reach `_emit_value`; blocks are in `_block_map` | — |

If any of these arrives at `_emit_value` *without* a `_value_map` entry, the assertion fires. Typical scenarios:

1. **Non-resource `Argument` not in map** (most common): an XIR pass (inlining, DCE, SROA) created or reparented an argument without re-registering it, or cbuffer setup in `_emit_kernel` missed an arg (e.g., `_property_ids.size() <= 2` with no fallback).
2. **`Instruction` result used before its defining instruction is emitted**: corrupted IR with a use-before-def cycle, or a PHI referencing a value from an unemitted block.
3. **`Function` (call target) not emitted**: dead code removal bug, or a call to a function never passed through `_emit_kernel`/`_emit_callable`.
4. **`BasicBlock` leaked as operand**: structural IR corruption — blocks should only be branch targets, never regular operands.

### Why This Assertion Is Necessary

The four types above cannot be materialized on-the-fly by `_emit_value` the way constants or builtins can. They depend on global state (function entry blocks, cbuffer layout, the block emission order). If `_emit_value` were to silently return `spv::NoResult`, the downstream `LUISA_ASSERT(id != spv::NoResult)` at line 327 would fire with "Failed to emit value" — a far less diagnostic message. This assertion catches the problem *at the point where the root cause is identifiable* (value type, name, type description), rather than later when a missing ID causes a cascade of failures.

### How to Solve

1. **Read the error message**: it prints `tag` (the `DerivedValueTag` string), `name`, and `type_desc`. This immediately tells you which kind of value is unregistered.
2. **For `ARGUMENT` (value/ref)**: check if the argument belongs to a kernel or callable. If kernel, verify cbuffer layout in `_emit_kernel` — ensure `_property_ids[2]` exists and the arg appears in `value_args`. If callable, check `_emit_callable` param mapping. If the IR was mutated by a pass, re-run that pass's registration logic or ensure the pass properly calls the emission entry points.
3. **For `INSTRUCTION`**: verify topological order. The instruction's defining block must have been emitted via `_emit_block` before its result is used. Check for PHI nodes that survived `reg2mem` (should be eliminated). Enable `LUISA_DUMP_SOURCE=1` to dump XIR at each pipeline stage.
4. **For `FUNCTION`**: ensure the callee was included in `used_functions_post_order` and emitted via `_emit_callable`. Check if `unused_callable_removal` pass incorrectly removed a callable that is still referenced.
5. **For `BASIC_BLOCK`**: this indicates a fundamental IR integrity issue. Dump XIR at the last pipeline stage, search for block values in instruction operand positions.
6. **General debugging**: set `LUISA_DUMP_SOURCE=1` and `LUISA_XIR_DISABLE_RESTRUCTURE_CFG=1` to narrow down which pipeline stage introduces the unregistered value. The XIR dump at each phase will show the IR state.

## Debug Workflow for Crashes / SPIR-V Validation Failures

When a test crashes or SPIR-V validation fails, follow this triage workflow:

### 1. Quick Triage: Disable All XIR Optimization

Set `LUISA_XIR_DISABLE_OPTIMIZATION=1` to skip the entire XIR optimization pipeline (Phase A, post-inline, CFG normalization). The raw AST→XIR module goes directly to SPIR-V codegen.

- **If the bug disappears**: the issue is in an XIR optimization pass. Re-enable optimization and bisect individual passes (use `LUISA_XIR_DISABLE_NORMALIZE_CFG=1` / `LUISA_XIR_DISABLE_RESTRUCTURE_CFG=1` to narrow further). See `xir_passes` skill for pass-level debugging.
- **If the bug persists**: the issue is in SPIR-V codegen or the Vulkan backend, not in XIR optimization. Proceed to Step 2 (disable SPIR-V optimizer).

### 2. Disable SPIR-V Optimizer & Other Passes

The optimization pipeline is defined in `src/backends/common/spirv/spirv_codegen/utils.cpp` inside `luisa_spirv_backend_translate_ast_to_xir()`. For finer-grained control beyond `LUISA_XIR_DISABLE_OPTIMIZATION`, temporarily comment out or skip individual passes:

- Phase A passes: `dce`, `local_store_forward`, `local_load_elimination`, `algebraic_simplify`, `const_fold`, `promote_ref_arg`, `sroa`, `loop_unroll`
- CFG normalization: `destructure_cfg`, `mem2reg`, `unused_callable_removal`, `simplify_cfg`
- CFG restructuring: `restructure_cfg`, `reg2mem`
- Phase B / Phase C passes after normalization / restructuring

Also disable the SPIR-V optimizer in `entry.cpp` (`compile_spirv()`): skip the `spvtools::Optimizer` `RegisterPerformancePasses()` run.

Re-run the test.

### 3. If Run Succeeds After Disabling Passes

The bug is introduced by an XIR pass. Narrow it down:

- Re-enable passes one group at a time (Phase A → CFG normalization → CFG restructuring → spv-opt).

- Re-enable passes one group at a time (Phase A → CFG normalization → CFG restructuring → spv-opt).
- Once the failing group is identified, bisect individual passes within that group.
- Debug the offending pass in `src/xir/passes/`. Common culprits:
  - `destructure_cfg` / `restructure_cfg`: corrupt structured control flow.
  - `mem2reg` / `reg2mem`: broken SSA / phi nodes.
  - `sroa`: incorrect scalar replacement of aggregates.
  - `loop_unroll`: unrolled loop body has invalid uses or dominates.
  - `unused_callable_removal`: removed a callable still referenced.
- Use `LUISA_DUMP_SOURCE=1` to compare XIR before / after the bad pass.

### 4. If Run Still Fails With All Passes Disabled

The bug is in the core SPIR-V codegen, not in optimizations. Start reading at `src/backends/common/spirv/spirv_codegen/entry.cpp` and follow the coding structure:

1. **`entry.cpp: compile_spirv()`** — verify `luisa_spirv_backend_translate_ast_to_xir()` produces a module.
2. **`bind.cpp: generate_binding()`** — check descriptor layout and global variable creation.
3. **`emit.cpp: emit()`** — verify type conversion, constant emission, and function emission order.
4. **`emit.cpp: _emit_kernel()`** — check cbuffer loading, entry point creation, dispatch bounds check.
5. **`emit.cpp: _emit_callable()`** — check parameter mapping and resource argument handling.
6. **`type.cpp: _convert_type()`** — verify the XIR type maps to valid SPIR-V type.
7. **`instruction.cpp: _emit_instruction()`** — locate the specific instruction that triggers the crash / validation error.
8. **`condition_inst.cpp`** — if the failure involves control flow, inspect if/loop/switch emission.

Set `LUISA_DUMP_SOURCE=1` to capture the final SPIR-V disassembly before the crash. If validation fails but codegen completes, run the SPIR-V binary through `spirv-val` directly to get the exact validation error and rule violated.

## Common SPIR-V Validation Errors

### Back-edge to non-loop-header block

**Error**: `Back-edges ('N[%N]' -> 'M[%M]') can only be formed between a block and a loop header`

**Root cause**: `restructure_cfg` creates a `SwitchInst` as part of multi-exit loop restructuring. A switch case body may contain a loop whose merge block branches back to the switch header. In structured SPIR-V, back-edges can only target blocks with `OpLoopMerge`, but the switch header has `OpSelectionMerge`.

**Typical XIR pattern**:
```
SwitchInst (header=H, merge=M)
  Case 1: ... 
    LoopInst (merge=L)
    L: BranchInst → H   ← back-edge to switch header!
  Default: M            ← switch merge (may contain return)
```

**Detection**: Before emitting the switch, traverse XIR successors from all case entry blocks. If any path reaches the switch header block (`inst->parent_block()`), the switch forms a loop and needs a wrapper.

**Fix in `_emit_switch_inst`** (`condition_inst.cpp`):

Wrap the switch in a synthetic SPIR-V loop:

```
loop_header (OpLoopMerge → loop_merge, loop_continue)
  → switch_spv_block (OpSelectionMerge → merge_block, OpSwitch)
  → cases ... → back-edge → loop_continue → loop_header
merge_block = switch merge (loop exit, may contain return)
loop_merge = unreachable (infinite loop exit)
```

Key implementation details:

1. **Block creation order matters**: `loop_header` must be created before all case blocks so the back-edge from case bodies to `loop_header` is lexically a back-edge (allowed since `loop_header` has `OpLoopMerge`).

2. **`_block_map` update**: After creating `loop_header`, update `_block_map[switch_xir_bb] = loop_header` so forward edges from outside the switch enter through the loop header.

3. **`_loop_header_redirect`**: Redirect branches that target the switch XIR header to `loop_continue` (the dedicated back-edge target block). This is set via `_loop_header_redirect.emplace(switch_xir_bb, loop_continue)`.

4. **Separate loop merge**: The loop merge must be a fresh block (`loop_merge`), NOT the switch merge (`merge_block`). A block cannot be the merge for two constructs (OpLoopMerge + OpSelectionMerge). Use `OpUnreachable` for the loop merge since the loop is infinite.

5. **Loop continue block**: Creates a dedicated `loop_continue` block that branches to `loop_header`. The back-edges from inside case bodies go to `loop_continue` via `_loop_header_redirect`.

**Debugging tips**:
- Dump the SPIR-V disassembly from the validation error message (add `spv::Disassemble` to `luisa_spirv_validate` temporarily).
- Identify the back-edge source and target blocks from the error message.
- Trace the block IDs to understand the XIR structure (switch header vs loop merge).
- Verify that the back-edge target is a switch header (`OpSelectionMerge`) not a loop header (`OpLoopMerge`).
- Check that the switch merge and loop merge are distinct blocks.


### GPU Crash Debugging Workflow (VK_ERROR_DEVICE_LOST)

When the Vulkan backend reports `VK_ERROR_DEVICE_LOST` at `vkQueueSubmit`, the GPU is crashing on previously submitted work. Key triage steps:

1. **Disable CFG passes first**: `LUISA_XIR_DISABLE_NORMALIZE_CFG=1` or `LUISA_XIR_DISABLE_RESTRUCTURE_CFG=1`. If the crash disappears, the bug is in CFG restructuring.
2. **Disable SPIR-V optimizer**: `LUISA_SPIRV_OPT_LEVEL=0`. The optimizer can inflate binary size (3831→5514 words, +43.9%) via inlining and may mask or introduce issues.
3. **Dump XIR at each stage**: `LUISA_DUMP_SOURCE=1` writes `.xir`, `.opt.xir`, `.norm.xir` files. Compare pre/post normalization to identify which pass corrupts the CFG.
4. **Compare with LLVM fallback**: The `fallback` backend (`luisa-backend-fallback`) uses its own XIR→LLVM codegen. If the fallback works but SPIR-V crashes, the bug is in SPIR-V codegen or the shared XIR passes.
5. **Check for infinite loops**: GPU timeout often means an infinite loop. Verify all loop back-edges are preserved after restructuring. Look for `restructured_loop` count mismatches.
6. **Check non-deterministic behavior**: `unordered_set` iteration order can cause intermittent bugs (e.g., `exit_targets[0]`). If a crash is flaky, suspect unordered container ordering.

## Coding Rules

These rules govern all code in `src/backends/common/spirv/spirv_codegen/`. They are derived from the existing implementation patterns and must be followed for consistency.

### File Organization

| Rule | Description |
|---|---|
| R1 | All codegen methods are declared in `entry.h` under `class SpirvCodegenEntry`. |
| R2 | `SpirvCodegenEntry` has only one public method: `static SpirvResult compile_spirv(...)`. Everything else is private. |
| R3 | Per-instruction dispatch lives in `instruction.cpp`. Control-flow (if/loop/switch/branch) lives in `condition_inst.cpp`. Type conversion lives in `type.cpp`. Binding/descriptor layout lives in `bind.cpp`. Core pipeline (emit/usage analysis/kernel prologue) lives in `emit.cpp`. |
| R4 | The AST→XIR translation and optimization pipeline is in `utils.cpp` — it is a free function `luisa_spirv_backend_translate_ast_to_xir()`, NOT a member of `SpirvCodegenEntry`. |

### Value Mapping (`_value_map`)

| Rule | Description |
|---|---|
| V1 | Every SPIR-V instruction result that is not void **must** be stored in `_value_map` via `_value_map.emplace(inst, id)`. |
| V2 | **EXCEPTION**: `SPECIAL_REGISTER` (builtins) results must **NOT** be cached in `_value_map`. Their `OpLoad` must dominate all uses; caching could place the load inside a loop body and reuse it after the loop, violating SPIR-V dominance. |
| V3 | **EXCEPTION**: `UNDEFINED` values must **NOT** be cached in `_value_map`. `OpUndef` is added to the current block and must dominate all uses. |
| V4 | `_emit_value()` handles cycle detection via `_emitting_values`. If a value is already being emitted (recursive cycle), an `OpUndef` placeholder is created and stored in `_value_map`. |
| V5 | If an instruction's parent block hasn't been emitted yet when `_emit_value()` is called for that instruction, the parent block is emitted on-the-fly (saving/restoring builder build point). |
| V6 | Non-resource `ARGUMENT` values are pre-mapped in `_emit_kernel()` (from cbuffer) or `_emit_callable()` (from function params). Resource `ARGUMENT` values are resolved lazily via `_resolve_resource_argument()`. |

### Type Conversion (`_convert_type`)

| Rule | Description |
|---|---|
| T1 | Always use `_convert_type(type, usage)` for SPIR-V result types. Usage is `Usage::READ` for read-only, `Usage::WRITE` for writable, `Usage::READ_WRITE` for read-write. |
| T2 | Texture types are cached separately in `_sampled_image_type_map` (read-only) vs `_storage_image_type_map` (writable). Do NOT cache them in `_type_map`. |
| T3 | Buffer types with struct/array elements use `_convert_laid_out_type()` which adds `ArrayStride`, `Offset`, `ColMajor`, `MatrixStride` decorations recursively. |
| T4 | Bool element types in buffers are always converted to `uint32` storage. Conversion back to bool happens on read (compare≠0) and on write (select 1:0). |
| T5 | FP8 types require `_uses_float8 = true`. Int8 types require `_uses_int8 = true`. 8-bit storage in StorageBuffer/Uniform/PushConstant requires tracking via `_mark_8bit_storage_usage()`. |
| T6 | New type tags: add a case in `_convert_type()` AND in `_convert_laid_out_type()` AND in `_emit_literal()` AND in `spirv_codegen_emit_scalar_constant()`. |

### Block Management

| Rule | Description |
|---|---|
| B1 | Blocks are created lazily via `_get_or_create_block(bb)` which checks `_block_map` first, then allocates a new `spv::Block` with `_builder.getUniqueId()`. |
| B2 | `_emit_block(bb)` returns `true` only when this call emitted the block body. It returns `false` for null, forwarded, already-emitted, and currently-emitting blocks. Treating `false` as an empty finished block can append instructions after a terminator when recursive emission later resumes. |
| B3 | Block emission order matters for SPIR-V dominance. Use `REVERSE_POST_ORDER` traversal in `_emit_function_blocks()` for forward functions. |
| B4 | For loops, create synthetic header/continue blocks **before** emitting body blocks so back-edges reference lexically earlier blocks (required by SPIR-V). |
| B5 | Merge blocks must be unique within a function — a block can only be the merge target for one construct. Use `_used_merge_blocks` to track. If a merge would be reused, create a synthetic merge block. |
| B6 | Pre-register all merge blocks via `_pre_register_merge_blocks(def)` before emitting any function body. This ensures block IDs are allocated before they are referenced as merge targets. |
| B7 | Use `_added_blocks` to track which SPIR-V blocks have been added to the function (via `addBlock()`). Do not add the same block twice. |
| B8 | Pending blocks (discovered via branch/cond_branch) are collected in `_pending_blocks` and emitted in FIFO order after the main loop body. This ensures definitions dominate uses. |

### Control Flow

| Rule | Description |
|---|---|
| C1 | All loop types (`LoopInst`, `SimpleLoopInst`, `RayQueryLoopInst`) use `_loop_header_redirect` to redirect branches to prepare/body/dispatch XIR blocks to the correct SPIR-V block (header or continue_block). |
| C2 | `_loop_boundary_stack` tracks (XIR merge block, SPIR-V merge/continue block) pairs per enclosing loop. `_conditional_branch_inst` uses this to detect break-vs-continue patterns and insert `OpSelectionMerge` when needed. |
| C3 | `_outer_merge_stack` tracks enclosing `IfInst` merge blocks so nested constructs can detect when an arm coincides with an outer merge — requiring a synthetic clone to avoid dominance ordering violations. |
| C4 | Switch instances must detect cross-segment branches (case bodies that branch to other cases or back to the switch header). If detected, wrap in a synthetic loop with `OpLoopMerge`. |
| C5 | `_emit_if_inst()`: if neither arm reaches the merge block (both exit via break/continue/return), the merge is unreachable. Use a synthetic dead-end merge; let the real merge be emitted when actually reached. |
| C6 | `OpSelectionMerge`/`OpLoopMerge` instructions must be emitted **before** the branch that targets the merge. |

### Constants, Literals, and the Constant UBO

| Rule | Description |
|---|---|
| K1 | `_emit_literal()` handles scalar, vector, matrix, array, and structure constants recursively. FP8 constants use `makeFloatE4M3Constant()` / `makeFloatE5M2Constant()`. |
| K2 | `_emit_constant()` checks `_value_map` cache first, then `_ubo_constant_member_by_hash` (if lowered to UBO), then falls back to `_emit_literal()`. |
| K3 | Array constants are detected in `compile_spirv()` (step after `_mark_atomic_buffer_types()`) and added to `_ubo_array_constants`. `generate_binding()` creates a `_ConstantUBO` with std140 layout. |
| K4 | UBO-lowered array constants use `_ubo_constant_member_by_hash` mapping constant hash → struct member index. `EXTRACT` on UBO arrays is detected in `_emit_arithmetic_inst()` for fast indexed load. |
| K5 | SpecConstantOp folding is enabled for `IAdd`, `IMul`, `IEqual`, `INotEqual` on integer constants. This is gated on `Op::OpConstant` or `Op::OpSpecConstant` opcodes for both operands. |

### Arithmetic Emission

| Rule | Description |
|---|---|
| A1 | The `_emit_arithmetic_inst()` switch must handle every `xir::ArithmeticOp`. Unhandled ops → `LUISA_NOT_IMPLEMENTED`. |
| A2 | Helper pattern: `unary(op)`, `binary(op)`, `glsl(builtin,args...)`, `glsl_typed(f,s,u,args...)`, and `make_glsl_call(builtin,type,args)`. |
| A3 | The `make_glsl_call` wrapper auto-promotes 8-bit integer operands to 32-bit before the GLSL.std.450 call and truncates the result back. |
| A4 | VectorTimesScalar peephole: in `BINARY_MUL` on float vectors, detect if one operand is a smeared scalar (`AGGREGATE` with all-equal operands, or load from alloca stored once from smeared scalar) and emit `OpVectorTimesScalar`. |
| A5 | Strength reduction: `BINARY_MUL(x, 2^n)` → `OpShiftLeftLogical(x, n)`. `BINARY_DIV(x, 2^n)` → `OpShiftRightArithmetic(x, n)` for signed int. |
| A6 | `AGGREGATE` peephole: detect when all operands are `EXTRACT` from the same base vector with constant indices → emit `createRvalueSwizzle`. |
| A7 | `SHUFFLE` peephole: if all indices are constants → `createRvalueSwizzle`. Otherwise → `createVectorExtractDynamic` per component. |
| A8 | `EXTRACT`: all-constant indices → `createCompositeExtract`. Vector with dynamic index → `createVectorExtractDynamic`. Array/matrix with small element count (≤16) and single dynamic index → `OpSelect` chain over `CompositeExtract`. Other dynamic cases → temp variable + access chain + load. |
| A9 | `INSERT`: all-constant indices → `createCompositeInsert`. Dynamic → temp variable + access chain + store + load. |
| A10 | FP8 types: **no arithmetic is allowed**. Emitters must error with guidance to upconvert to float16/float32, compute, then downconvert. |
| A11 | Bool arithmetic maps to logical ops: `BINARY_BIT_AND`→`OpLogicalAnd`, `BINARY_BIT_OR`→`OpLogicalOr`, `BINARY_BIT_XOR`→`OpLogicalNotEqual`, `UNARY_BIT_NOT`→`OpLogicalNot`. Comparison ops use `OpFOrd*` for float, `OpS*` for signed int, `OpU*` for unsigned. |

### Resource Read/Write

| Rule | Description |
|---|---|
| W1 | Typed buffer access (when buffer element type is non-null and not word-storage): use `_create_access_chain(0, index)` → `createLoad`/`createStore`. Type mismatch handled with `OpCopyLogical`. |
| W2 | Byte buffer / untyped / bindless access: compute `word_offset = index * word_count`, use `_emit_buffer_read_impl()`/`_emit_buffer_write_impl()` recursively. |
| W3 | Sub-word types (bool, int8, float16, float8, sub-word structures/vectors) in byte buffers use read-modify-write: load word, shift, mask, convert, and for writes: clear old bits, OR new bits. |
| W4 | `_emit_buffer_read_impl()` and `_emit_buffer_write_impl()` recursively handle scalar, vector, matrix, structure, and array element types. Multi-word scalars (int64, double) are loaded/stored as uvec2 via bitcast. |
| W5 | Bool vector storage: reads extract individual bits via shift+mask+compare; writes pack bits via select+shift+or. |
| W6 | Bindless buffer access: resolve `buffer_idx` from bindless_array (3 uint32s per slot), create access chain into `_buffer_heap_id`. Mark `NonUniformEXT` decoration when the slot index is non-uniform. |
| W7 | Textures: use `_load_texture()` to handle array-vs-non-array image loads. Storage images use `OpImageRead`/`OpImageWrite` (+`StorageImageReadWithoutFormat`/`StorageImageWriteWithoutFormat` capabilities). Sampled images use `OpImageFetch`/`createTextureCall`. |

### Atomic Emission

| Rule | Description |
|---|---|
| AT1 | Scope is always `Device`. Memory semantics are `MaskNone` (no acquire/release needed for compute shaders). |
| AT2 | Float atomics on scalar buffers use native SPIR-V extensions: `SPV_EXT_shader_atomic_float_add` for add, `SPV_EXT_shader_atomic_float_min_max` for min/max, `SPV_EXT_shader_atomic_float16_add` for half. |
| AT3 | Float atomics on non-scalar buffers (word-storage) fall back to CAS loop: `OpAtomicLoad` → compute new value → `OpAtomicCompareExchange` → loop until success. This is required because NVIDIA drivers crash on pointer bitcast in atomics. |
| AT4 | Float compare-exchange on scalar buffers: CAS loop with `OpAtomicLoad` + `OpAtomicExchange` (avoids `OpAtomicCompareExchange` on float pointers). |
| AT5 | Non-scalar buffer atomics compute word offsets from element indices: `elem_index * elem_word_count + byte_offset/4`. |
| AT6 | `_needs_atomic_buffer_types` tracks buffer types that require atomic access → influences `_buffer_uses_word_storage()` which determines whether the buffer uses uint32 storage (word-level) or typed access. |

### Kernel and Callable Emission

| Rule | Description |
|---|---|
| E1 | Kernel entry point must have **no parameters** in SPIR-V. Use `makeFunctionEntry` with empty param lists. |
| E2 | Non-resource kernel arguments are loaded from cbuffer (`_property_ids[2]`): compute aligned offsets, handle sub-word types (bool→compare, int8→truncate+bitcast), multi-word types via `_emit_buffer_read_impl`. |
| E3 | If cbuffer is unavailable (`_property_ids.size() <= 2`), fall back to `makeNullConstant` for value args. |
| E4 | Kernel prologue must include **dispatch bounds check**: `GlobalInvocationId >= DispatchSize` → early return. This prevents extra threads in the last workgroup from executing. Uses `OpSelectionMerge` + `OpBranchConditional` → return block / body block. The XIR body block is remapped to the body block (not the entry). |
| E5 | Callable emission: skip unused resource args (to avoid type mismatches with kernel globals). Add `VariablePointersStorageBuffer` capability when callable has buffer/bindless_array params. |
| E6 | Callable texture params are marked in `_is_storage_image_map` using `_function_argument_usage_of()`. |
| E7 | `_callable_arg_used` tracks which callable args are actually used (via `use_list().empty()`). Unused resource args are skipped during call site argument packing. |
| E8 | Call site (`CALL`): for reference args that are not alloca/param, create temp variable, copy-in before call, copy-out after call. |
| E9 | Before emitting each function (`_emit_kernel`/`_emit_callable`), call `_reset_function_codegen_state()` to clear per-function state. |
| E10 | After emitting all blocks, ensure the last block is terminated. If not, add `makeReturn(false)`. |

### Special Registers and Builtins

| Rule | Description |
|---|---|
| S1 | Built-in variables are created on first use and cached in `_builtin_var_map`. They use `Input` storage class with `BuiltIn` decoration. |
| S2 | `DISPATCH_SIZE` is a special case: it reads from push constant `_property_ids[0]` (uint4), extracting xyz components. |
| S3 | `_global_invocation_id_var` is cached separately because it's needed in the kernel prologue (dispatch bounds check). If not yet created, it's created there. |
| S4 | Builtin map: `THREAD_ID`→LocalInvocationId, `BLOCK_ID`→WorkgroupId, `DISPATCH_ID`→GlobalInvocationId, `BLOCK_SIZE`→WorkgroupSize, `WARP_SIZE`→SubgroupSize, `WARP_LANE_ID`→SubgroupLocalInvocationId. |

### Capabilities and Extensions

| Rule | Description |
|---|---|
| CAP1 | Base setup in constructor: `spv::Spv_1_5`, `Shader` capability, `GLSL450` memory model, import `GLSL.std.450`. |
| CAP2 | Image read/write without format: `StorageImageReadWithoutFormat` / `StorageImageWriteWithoutFormat` are core SPIR-V 1.5 capabilities (no extension needed). |
| CAP3 | Image query: `ImageQuery` capability required for `OpImageQuerySize`/`OpImageQuerySizeLod`. |
| CAP4 | 8-bit storage: `SPV_KHR_8bit_storage` extension + `StorageBuffer8BitAccess` / `UniformAndStorageBuffer8BitAccess` / `StoragePushConstant8`. |
| CAP5 | Descriptor indexing: `SPV_EXT_descriptor_indexing` + `RuntimeDescriptorArray` + `ShaderNonUniformEXT` + array-specific indexing caps (`SampledImageArrayNonUniformIndexingEXT`, `StorageImageArrayNonUniformIndexingEXT`, `StorageBufferArrayNonUniformIndexingEXT`). |
| CAP6 | Ray query: `SPV_KHR_ray_query` + `RayQueryKHR`. |
| CAP7 | Variable pointers: `SPV_KHR_variable_pointers` + `VariablePointersStorageBuffer` (only for callables with buffer/bindless_array params). |
| CAP8 | Group ops: `GroupNonUniform` + `GroupNonUniformArithmetic`/`GroupNonUniformVote`/`GroupNonUniformBallot`/`GroupNonUniformShuffle` as needed. |
| CAP9 | Add capabilities and extensions as late as possible (at emission time), not in the constructor, so they are only added when actually needed. |

### SPIR-V Validator Interaction

| Rule | Description |
|---|---|
| VAL1 | SPIR-V validation runs twice: pre-optimization and post-optimization. Environment: `SPV_ENV_VULKAN_1_2`. |
| VAL2 | Validation failure → `LUISA_ERROR` with stage ("pre-optimization"/"post-optimization") and full message log. |
| VAL3 | SPIR-V optimization level controlled by `LUISA_SPIRV_OPT_LEVEL` env var: 0=skip, 1=lightweight (ADCE+BlockMerge+Simplification+DeadBranchElim), 2=performance passes. |
| VAL4 | The optimizer is an `spvtools::Optimizer` run on the SPIR-V binary after initial dump and pre-validation. |

### Lifecycle and Cleanup

| Rule | Description |
|---|---|
| L1 | `SpirvCodegenEntry` is constructed once per `compile_spirv()` call. Constructor creates `spv::Builder` with `spv::Spv_1_5`. |
| L2 | The `spv::Builder` is intentionally leaked via `_builder_ptr.release()` in `compile_spirv()` to avoid destructor crash. This is a known glslang issue. |
| L3 | Destructor clears all maps and containers explicitly (though rarely called due to leak). |
| L4 | `compile_spirv()` is the only public API. It is called from `vk/device.cpp` (Vulkan backend) at shader creation time. |

### General Coding Patterns

| Rule | Description |
|---|---|
| G1 | Use `luisa::` containers (`luisa::vector`, `luisa::unordered_map`, `luisa::unordered_set`, `luisa::string`, `luisa::span`, `luisa::unique_ptr`). |
| G2 | Use `spv::NoResult` as the null/invalid SPIR-V ID sentinel. Use `spv::StorageClass::Max` as uninitialized storage class sentinel. |
| G3 | All emission methods return `void` and store results in `_value_map` (except `_emit_value`, `_emit_constant`, `_emit_literal`, `_resolve_resource_argument`, `_emit_buffer_read*`, `_ensure_type` which return `spv::Id`). |
| G4 | Use `LUISA_ASSERT(id != spv::NoResult, ...)` after every emission to catch failures early. |
| G5 | Use `LUISA_NOT_IMPLEMENTED(...)` for unhandled opcodes/types — this produces a clear error message during shader compilation. |
| G6 | The pattern for emitting a new instruction type: (1) add case in `_emit_instruction()`, (2) implement the emission method in `instruction.cpp` (or `condition_inst.cpp` for control flow), (3) declare it in `entry.h`. |
| G7 | Access chains: always use `_create_access_chain()` which saves/restores the builder's AccessChain state. This prevents cross-contamination between access chains. |
| G8 | Type matching: use `_ensure_type()` when operands may need implicit conversion (e.g., uint→int for combined arithmetic). Falls back to `OpBitcast` when no better conversion exists. |
| G9 | Builder build point management: when emitting blocks out-of-order (e.g., in `_emit_value`), always save `_builder.getBuildPoint()` first and restore it after. |
| G10 | `_uniformity.analyze(f)` must be called at the start of `_emit_kernel()` and `_emit_callable()` so uniformity information is available for `NonUniformEXT` decoration decisions. |
| G11 | `_function_argument_usage` is computed in `_analyze_function_argument_usage()` before function emission. It propagates usage (READ/WRITE/READ_WRITE) through call chains with a fixed-point iteration. |
| G12 | When adding a new env var for debugging: define it as a file-local `const bool` in `utils.cpp` (read once via `getenv`), check it in the appropriate pipeline section, and document it in this skill file.
