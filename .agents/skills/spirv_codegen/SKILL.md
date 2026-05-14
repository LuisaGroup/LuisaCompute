---
name: spirv_codegen
description: SPIR-V backend codegen — XIR→SPIR-V via glslang Builder, type/instruction/control-flow/binding emission, and debugging
---

# SPIR-V Codegen

Translates XIR→SPIR-V using glslang's `spv::Builder`. Located in `src/backends/common/spirv/spirv_codegen/`.

## Directory Layout

```
src/backends/common/spirv/spirv_codegen/
├── lc_spirv_pch.h       # PCH: all luisa/xir headers + SPIRV/SpvBuilder.h
├── entry.h/cpp          # SpirvCodegenEntry, compile_spirv() entry point
├── emit.cpp             # Usage analysis, block/function/kernel emission
├── type.cpp             # Type→spv::Id conversion
├── instruction.cpp      # Per-instruction: arithmetic, atomic, resource, thread_group
├── condition_inst.cpp   # Control flow: if, loop, switch, branch
├── bind.cpp             # Descriptor bindings, global variables, decorations
├── property.h           # Re-exports hlsl::Property / ShaderVariableType
└── utils.h/cpp          # AST→XIR translation helper
```

Build: xmake target `lc-spirv` (static lib `luisa-spirv.lib`), deps `lc-vstl`, `lc-runtime`, `lc-glslang`.

## Architecture

### Two-Phase Emission

1. **Analysis** (`_analyze_instruction_usage`): post-order traversal collecting `used_types`, `used_constants`, `used_functions_post_order`, print info.
2. **Emission** (`emit()`): convert types → emit constants → emit functions → `postProcess(false)` → `dump()`.

### Key Class

```cpp
class SpirvCodegenEntry {
    spv::Builder _builder;
    spv::SpvBuildLogger _logger;
    unordered_map<const Type*, spv::Id> _type_map;
    unordered_map<const xir::Value*, spv::Id> _value_map;
    unordered_map<const xir::Function*, spv::Function*> _function_map;
    unordered_map<const xir::BasicBlock*, spv::Block*> _block_map;
    unordered_map<const xir::BasicBlock*, pair<spv::Block*, spv::Block*>> _loop_header_info;
    // → maps loop prepare block to (merge, update)
    unordered_set<const xir::BasicBlock*> _emitted_blocks;
    unordered_map<spv::Id, bool> _is_storage_image_map;

    SpirvResult::Properties _properties;
    vector<spv::Id> _property_ids;
    bool _use_tex2d_bindless, _use_tex3d_bindless, _use_buffer_bindless;
    spv::Id _glsl450;
};
```

### Entry Point

```cpp
SpirvResult compile_spirv(Function kernel, const ShaderOption &opt) {
    auto xir_module = luisa_spirv_backend_translate_ast_to_xir(kernel, opt);
    SpirvCodegenEntry codegen{scratch, true/*indirect*/};
    codegen.generate_binding(kernel);
    codegen.emit(xir_module.get(), kernel.bound_arguments(), {}, opt.native_include);
    codegen._builder.dump(words);
    return SpirvResult{words, properties, printers, tex2d, tex3d, buffer};
}
```

## Type Conversion (`type.cpp`)

`_convert_type()` — cached in `_type_map`:

| Type Tag | SPIR-V Call |
|---|---|
| BOOL/INT*/UINT*/FLOAT* | `makeBoolType()` / `makeIntType(w)` / `makeUintType(w)` / `makeFloatType(w)` |
| VECTOR | `makeVectorType(elem, dim)` |
| MATRIX | `makeMatrixType(elem, dim, dim)` |
| ARRAY | `makeArrayType(elem, size_id, 0)` |
| STRUCTURE | `makeStructType(member_types, {}, "Struct", false)` |
| Buffer/Texture/Bindless/Accel/Custom | `LUISA_NOT_IMPLEMENTED` (handled as bindings) |

## Value Emission (`emit.cpp`)

```cpp
spv::Id _emit_constant(const xir::Constant *c);   // scalar→make*Constant, vector→composite
spv::Id _emit_value(const xir::Value *value);      // dispatches by DerivedValueTag
```

| Tag | Handling |
|---|---|
| CONSTANT | `_emit_constant()` |
| UNDEFINED | `createUndefined(type)` |
| SPECIAL_REGISTER | Input var + BuiltIn decoration + createLoad |
| ARGUMENT (resource) | `_resolve_resource_argument()` → `_property_ids` lookup |
| ARGUMENT (value) | Function param ID from kernel/callable emission |
| FUNCTION/BLOCK/INSTRUCTION | Pre-mapped in `_value_map`/`_block_map`/`_function_map` |

Special registers: LocalInvocationId, WorkgroupId, GlobalInvocationId, WorkgroupSize, NumWorkgroups, SubgroupSize, SubgroupLocalInvocationId.

## Instruction Emission (`instruction.cpp`)

### Arithmetic (`_emit_arithmetic_inst`)
- Lambdas: `unary(op)`, `binary(op)`, `glsl(builtin,...)`, `glsl_typed(f,s,u,...)`
- Maps `ArithmeticOp`→`spv::Op` or `GLSLstd450*`
- Boolean reductions: `OpAll`, `OpAny`
- Vector reductions (sum/product/min/max): manual extract + per-element combine
- Matrix: `OpFNegate` per row, `OpMatrixTimesScalar/Vector/Matrix`, `OpTranspose`, `GLSLstd450Determinant/MatrixInverse`
- Composite: `OpCompositeConstruct/Extract/Insert`, `OpVectorExtractDynamic` (shuffle)

### Atomic (`_emit_atomic_inst`)
Scope=Device, Semantics=MaskNone. Base pointer + optional `createAccessChain`. Maps to `OpAtomicExchange/CompareExchange/IAdd/ISub/And/Or/Xor/SMin/UMin/SMax/UMax`.

### Resource (`_emit_resource_query/read/write_inst`)
- Buffer: `createArrayLength`+`OpIMul` for byte size; `OpBitcast` from uint32
- Texture: `OpImageQuerySize/QuerySizeLod` (+Capability::ImageQuery)
- Read: `OpImageRead` (storage) / `OpImageFetch` (sampled) + `StorageImageReadWithoutFormat`
- Write: `OpImageWrite` + `StorageImageWriteWithoutFormat`
- Buffer r/w: `createAccessChain` into StorageBuffer + `createLoad`/`createStore`

### Thread Group (`_emit_thread_group_inst`)
- SYNCHRONIZE_BLOCK: `createControlBarrier(Workgroup, Workgroup, AcquireRelease)`
- Warp: `OpGroupNonUniformElect/All/Any/BitwiseAnd/Or/Xor` with `GroupOperation::Reduce` (+Capability::GroupNonUniform)

### Generic Dispatch (`_emit_instruction`)
```cpp
ALLOCA→createVariable(Function)  LOAD→createLoad  STORE→createStore  GEP→createAccessChain
ARITHMETIC→_emit_arithmetic_inst  CALL→createFunctionCall (skip resource args)
CAST→OpBitcast/OpSelect/OpConvert*/OpFConvert/OpSConvert/OpUConvert
IF/LOOP/SIMPLE_LOOP/BRANCH/CONDITIONAL_BRANCH/BREAK/CONTINUE/RETURN→condition_inst.cpp
UNREACHABLE→OpUnreachable  ATOMIC→_emit_atomic_inst  RESOURCE_*→_emit_resource_*_inst
THREAD_GROUP→_emit_thread_group_inst
PHI→ERROR (must be eliminated)  RAY_QUERY_*/AUTODIFF_*/etc.→ERROR or NOT_IMPLEMENTED
// NOT_IMPLEMENTED: PRINT, CLOCK, ASSERT, ASSUME, DEBUG_BREAK, OUTLINE, RASTER_DISCARD
```

## Control Flow (`condition_inst.cpp`)

- **If**: create true/false/merge blocks, emit `OpSelectionMerge`+`OpBranchConditional`, then emit each block.
- **Loop** (`_emit_loop_inst`): `createBranch(prepare)` → emit prepare/body/update/merge. `_loop_header_info[prepare] = (merge, update)` so `_emit_block` emits `OpLoopMerge` at header.
- **Simple Loop** (`_emit_simple_loop_inst`): do-while, body as header: `_loop_header_info[body] = (merge, body)`.
- **Branch/CondBranch**: direct `createBranch`/`createConditionalBranch`, then recursively `_emit_block` targets.

## Binding Generation (`bind.cpp`)

`generate_binding(Function kernel)` inspects arguments + builtin callables → descriptor layout:

| Index | Type | SPIR-V Global |
|---|---|---|
| 0 | ConstantValue | PushConstant struct (uint4), space=0 reg=0 |
| 1 | SamplerHeap | UniformConstant sampler array, space=1 reg=0, size=16 |
| 2 | StructuredBuffer (opt) | CBuffer for non-resource args |
| ... | SRVBufferHeap | bindless buffers (if `_use_buffer_bindless`) |
| ... | SRVTextureHeap / UAVTextureHeap | bindless 2D/3D textures |
| ... | Per-arg textures/buffers/accel | individual bindings, space=0 |
| ... | RWStructuredBuffer x2 | print buffers (if `requires_printing()`) |

Property→SPIR-V mapping: PushConstant (Block), SamplerHeap (UniformConstant array), StructuredBuffer (StorageBuffer+Block+RuntimeArray), SRV/UAVTextureHeap (UniformConstant image array, Dim2D, sampled=1/2), SRVBufferHeap (StorageBuffer+Block), SPIRVAccel (UniformConstant AccelerationStructure).

Resource args resolved via `_resolve_resource_argument()`: computes `_property_ids` index = position + base offset (accounts for cbuffer, bindless flags).

## Kernel vs Callable (`emit.cpp`)

- **Kernel**: void return, non-resource params only. Entry: `addEntryPoint(GLCompute, func, "main")`, mode: `LocalSize(block_size)`. Maps param IDs→`xir::Argument`.
- **Callable**: return type from `_convert_type(callable->type())`, non-resource params, name from callable metadata. No entry point/mode.

## Block Emission

```cpp
void _emit_block(const xir::BasicBlock *bb) {
    if (!bb || emitted) return;
    auto spv_block = _get_or_create_block(bb);
    _builder.setBuildPoint(spv_block);
    if (bb is loop header) _builder.createLoopMerge(merge, continue_target, MaskNone);
    for (auto inst : bb->instructions()) _emit_instruction(inst);
}
```

## Adding New Instructions

1. **Arithmetic op**: add case in `_emit_arithmetic_inst`, map to `spv::Op`/`GLSLstd450*`
2. **Resource op**: add to `_emit_resource_read/write/query_inst`
3. **Thread-group op**: add to `_emit_thread_group_inst`
4. **Control flow**: add method in `entry.h`, implement in `condition_inst.cpp`, dispatch from `_emit_instruction`
5. **Binding support**: update `generate_binding()` in `bind.cpp`
6. **New type**: update `_convert_type()` in `type.cpp`

All paths must: use `_convert_type(inst->type())` for result, `_emit_value(op)` for operands, store in `_value_map` (or `spv::NoResult` for void).

## Result & Debugging

```cpp
struct SpirvResult {
    vector<uint32_t> spv_bin;
    vector<Property> properties;  // consumed by Vulkan descriptor allocator
    vector<pair<string, const Type*>> printers;  // host-side print formatting
    bool useTex2DBindless, useTex3DBindless, useBufferBindless;
};
```

### `LUISA_DUMP_SOURCE=1`
Set env var to dump codegen results. In `Device::create_shader()`:

- **XIR→SPIRV path** (`LUISA_XIR_TO_SPIRV`): logs binary size + property bindings. If `print_code()`: writes `spirv_output.spvasm` (XIR→SPIRV disassembly) + `spirv_output_hlsl.spvasm` (HLSL→DXC→SPIRV for comparison). HLSL code writes to `hlsl_output.hlsl`
- **HLSL-only path**: if `compile_only` + `print_code()`: writes `hlsl_output.hlsl`.

```bash
export LUISA_DUMP_SOURCE=1   # or set/=$env:
```

When `LUISA_XIR_TO_SPIRV` is undefined, backend falls back to HLSL codegen + DXC.
