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
- **Branch** (`_emit_branch_inst`): respects `_loop_header_redirect` for target.
- **CondBranch** (`_emit_conditional_branch_inst`): respects `_loop_header_redirect` for both targets.

Key patterns:
- `_used_merge_blocks`: prevents reusing merge block IDs (SPIR-V requires unique merge targets).
- `_loop_header_redirect`: For all loop types, redirects branches to prepare/body/dispatch to the correct SPIR-V block (header or continue_block).
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

- **XIR→SPIRV path** (`LUISA_XIR_TO_SPIRV`): logs binary size + property bindings. `print_code()`: writes `spirv_output.spvasm` (XIR→SPIRV disassembly) + `spirv_output_hlsl.spvasm` (HLSL→DXC→SPIRV for comparison). HLSL writes to `hlsl_output.hlsl`.
- **HLSL-only path**: `compile_only` + `print_code()`: writes `hlsl_output.hlsl`.

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
