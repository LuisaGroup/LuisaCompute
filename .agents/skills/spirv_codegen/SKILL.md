# SPIR-V Codegen Skill

SPIR-V backend code generator for LuisaCompute. Translates XIR (extended IR) to SPIR-V binary using glslang's `spv::Builder`. Located in `src/backends/common/spirv/spirv_codegen/`.

## Directory Layout

```
src/backends/common/spirv/
├── spirv_codegen/
│   ├── lc_spirv_pch.h          # Precompiled header: all luisa/xir headers + SPIRV/SpvBuilder.h
│   ├── entry.h / entry.cpp     # SpirvCodegenEntry public API, compile_spirv() entry point
│   ├── emit.cpp                # Constructor, usage analysis, block/function/kernel emission
│   ├── type.cpp                # luisa::compute::Type -> spv::Id conversion
│   ├── instruction.cpp         # Per-instruction emission: arithmetic, atomic, resource, thread_group, generic ops
│   ├── condition_inst.cpp      # Control flow: if, loop, simple_loop, switch, branch, conditional_branch
│   ├── bind.cpp                # Descriptor binding generation, SPIR-V global variables, decorations
│   ├── property.h              # Re-exports hlsl::Property / ShaderVariableType
│   └── utils.h / utils.cpp     # AST-to-XIR translation helper: luisa_spirv_backend_translate_ast_to_xir()
└── xmake.lua                   # Builds static target "lc-spirv" (luisa-spirv.lib)
```

## Build System

```lua
-- src/backends/common/spirv/xmake.lua
target("lc-spirv")
    set_basename("luisa-spirv")
    add_deps("lc-vstl", "lc-runtime", "lc-glslang")
    add_files("spirv_codegen/*.cpp")
    lc_set_pcxxheader("spirv_codegen/lc_spirv_pch.h")
```

Depends on glslang's SPIRV builder (`SPIRV/SpvBuilder.h`). Uses `-fms-extensions` on non-Windows Clang.

## Architecture Overview

### Two-Phase Emission

1. **Analysis phase** (`_analyze_instruction_usage`): traverses kernel and all reachable callables post-order, collecting:
   - `used_types` — all `Type*` that need SPIR-V type declarations
   - `used_constants` — all `xir::Constant*` to emit as SPIR-V constants
   - `used_functions_post_order` — callable before kernel, for forward declarations
   - `_requires_printing` / `_print_info` — print buffer metadata

2. **Emission phase** (`emit()`):
   - `_convert_type()` for all used types
   - `_emit_constant()` for all used constants
   - `_emit_kernel()` / `_emit_callable()` for all functions
   - `_builder.postProcess(false)` then `_builder.dump()`

### Key Class: `SpirvCodegenEntry`

```cpp
namespace lc::spirv {
class SpirvCodegenEntry {
    StringScratch &_scratch;
    spv::Builder _builder;
    spv::SpvBuildLogger _logger;

    // XIR -> SPIR-V maps
    luisa::unordered_map<const Type *, spv::Id> _type_map;
    luisa::unordered_map<const xir::Value *, spv::Id> _value_map;
    luisa::unordered_map<const xir::Function *, spv::Function *> _function_map;
    luisa::unordered_map<const xir::BasicBlock *, spv::Block *> _block_map;
    luisa::unordered_map<const xir::BasicBlock *, std::pair<spv::Block *, spv::Block *>> _loop_header_info;
    luisa::unordered_set<const xir::BasicBlock *> _emitted_blocks;

    // Binding / property state
    SpirvResult::Properties _properties;
    luisa::vector<spv::Id> _property_ids;
    bool _use_tex2d_bindless{false};
    bool _use_tex3d_bindless{false};
    bool _use_buffer_bindless{false};
    spv::Id _glsl450{spv::NoResult};
    luisa::unordered_map<spv::Id, bool> _is_storage_image_map;
};
}
```

### Entry Point Flow

```cpp
SpirvResult SpirvCodegenEntry::compile_spirv(Function kernel, const ShaderOption &opt) {
    // 1. AST -> XIR
    auto xir_module = luisa::compute::spirv::luisa_spirv_backend_translate_ast_to_xir(kernel, opt);

    // 2. Setup codegen
    SpirvCodegenEntry codegen{scratch, /*allow_indirect=*/true};
    codegen.generate_binding(kernel);

    // 3. Emit SPIR-V
    codegen.emit(xir_module.get(), kernel.bound_arguments(), {}, opt.native_include);

    // 4. Extract binary + properties
    codegen._builder.dump(words);
    return SpirvResult{words, properties, printers, tex2d, tex3d, buffer};
}
```

## Type Conversion (`type.cpp`)

Maps `luisa::compute::Type` tags to `spv::Builder` type creation. Cached in `_type_map`.

```cpp
spv::Id SpirvCodegenEntry::_convert_type(const Type *type) noexcept;
```

| Type Tag | SPIR-V Builder Call |
|----------|---------------------|
| `BOOL` | `makeBoolType()` |
| `INT8/16/32/64` | `makeIntType(w)` |
| `UINT8/16/32/64` | `makeUintType(w)` |
| `FLOAT16/32/64` | `makeFloatType(w)` |
| `VECTOR` | `makeVectorType(elem, dim)` |
| `MATRIX` | `makeMatrixType(elem, dim, dim)` |
| `ARRAY` | `makeArrayType(elem, size_id, 0)` |
| `STRUCTURE` | `makeStructType(member_types, {}, "Struct", false)` |
| Buffer/Texture/Bindless/Accel/Custom | `LUISA_NOT_IMPLEMENTED` (handled as bindings, not value types) |

## Constant & Value Emission (`emit.cpp`)

### Constants

```cpp
spv::Id SpirvCodegenEntry::_emit_constant(const xir::Constant *c) noexcept;
```

- Scalar: direct `make*Constant` from `c->data()`
- Vector: per-element scalar constants + `makeCompositeConstant`
- Cached in `_value_map`

### Generic Values

```cpp
spv::Id SpirvCodegenEntry::_emit_value(const xir::Value *value) noexcept;
```

| `DerivedValueTag` | Handling |
|-------------------|----------|
| `CONSTANT` | `_emit_constant()` |
| `UNDEFINED` | `createUndefined(type)` |
| `SPECIAL_REGISTER` | Input variable + `BuiltIn` decoration + `createLoad` (LocalInvocationId, WorkgroupId, GlobalInvocationId, WorkgroupSize, NumWorkgroups, SubgroupSize, SubgroupLocalInvocationId) |
| `ARGUMENT` (resource) | `_resolve_resource_argument()` (looks up `_property_ids`) |
| `ARGUMENT` (value) | Function parameter ID mapped during `_emit_kernel` / `_emit_callable` |
| `FUNCTION` / `BASIC_BLOCK` / `INSTRUCTION` | Must be pre-mapped in `_value_map` / `_block_map` / `_function_map` |

## Instruction Emission (`instruction.cpp`)

### Arithmetic (`_emit_arithmetic_inst`)

- Uses lambdas `unary(op)`, `binary(op)`, `glsl(builtin, ...)`, `glsl_typed(f, s, u, ...)`
- Maps `xir::ArithmeticOp` to `spv::Op` or `GLSL.std.450` builtins
- Boolean reductions: `OpAll`, `OpAny`
- Vector reductions (sum/product/min/max): manual composite extract + per-element combine
- Matrix ops: `OpFNegate` per row, `OpMatrixTimesScalar`, `OpVectorTimesMatrix`, `OpMatrixTimesVector`, `OpMatrixTimesMatrix`, `OpTranspose`, `GLSLstd450Determinant` / `MatrixInverse`
- Composite: `OpCompositeConstruct`, `OpCompositeExtract`, `OpCompositeInsert`, `OpVectorExtractDynamic` (shuffle)

### Atomic (`_emit_atomic_inst`)

- Base pointer + optional `createAccessChain` for indices
- Scope=`Device`, Semantics=`MaskNone`
- Maps to `OpAtomicExchange`, `OpAtomicCompareExchange`, `OpAtomicIAdd`, `OpAtomicISub`, `OpAtomicAnd`, `OpAtomicOr`, `OpAtomicXor`, `OpAtomicSMin`, `OpAtomicUMin`, `OpAtomicSMax`, `OpAtomicUMax`

### Resource Query / Read / Write (`_emit_resource_query_inst`, `_emit_resource_read_inst`, `_emit_resource_write_inst`)

- Buffer: `createArrayLength` + `OpIMul` for byte size; `OpBitcast` from uint32 raw words
- Texture: `OpImageQuerySize` / `OpImageQuerySizeLod` (adds `Capability::ImageQuery`)
- Texture read: `OpImageRead` (storage) / `OpImageFetch` (sampled); adds `StorageImageReadWithoutFormat`
- Texture write: `OpImageWrite`; adds `StorageImageWriteWithoutFormat`
- Buffer read/write: `createAccessChain` into `StorageBuffer` + `createLoad` / `createStore`

### Thread Group (`_emit_thread_group_inst`)

- `SYNCHRONIZE_BLOCK`: `createControlBarrier(Workgroup, Workgroup, AcquireRelease)`
- Warp ops: `OpGroupNonUniformElect`, `OpGroupNonUniformAll`, `OpGroupNonUniformAny`, `OpGroupNonUniformBitwiseAnd/Or/Xor` with `GroupOperation::Reduce`; adds `Capability::GroupNonUniform`

### Generic Dispatcher (`_emit_instruction`)

```cpp
switch (inst->derived_instruction_tag()) {
    case ALLOCA:   createVariable(Function storage class)
    case LOAD:     createLoad
    case STORE:    createStore
    case GEP:      createAccessChain
    case ARITHMETIC:   _emit_arithmetic_inst
    case CALL:     createFunctionCall (skips resource args)
    case CAST:     OpBitcast / OpSelect / OpConvert* / OpFConvert / OpSConvert / OpUConvert
    case IF:       _emit_if_inst
    case LOOP:     _emit_loop_inst
    case SIMPLE_LOOP: _emit_simple_loop_inst
    case BRANCH:   createBranch
    case CONDITIONAL_BRANCH: createConditionalBranch
    case BREAK:    createBranch to target
    case CONTINUE: createBranch to target
    case RETURN:   makeReturn
    case UNREACHABLE: OpUnreachable
    case ATOMIC:   _emit_atomic_inst
    case RESOURCE_QUERY:  _emit_resource_query_inst
    case RESOURCE_READ:   _emit_resource_read_inst
    case RESOURCE_WRITE:  _emit_resource_write_inst
    case THREAD_GROUP:    _emit_thread_group_inst
    case PHI:      ERROR (must be eliminated before codegen)
    case RAY_QUERY_LOOP / AUTODIFF_SCOPE / ...: ERROR (must be eliminated)
    case PRINT / CLOCK / ASSERT / ASSUME / DEBUG_BREAK / OUTLINE / RASTER_DISCARD / RAY_QUERY_*:
        NOT_IMPLEMENTED
}
```

## Control Flow (`condition_inst.cpp`)

### If-Then-Else

Creates `true_block`, `false_block`, `merge_block` as new `spv::Block`s, emits `OpSelectionMerge` + `OpBranchConditional`, then emits true/false/merge blocks with fall-through branches.

### Loop (`_emit_loop_inst`)

```
builder.createBranch(prepare);
_emit_block(prepare);  // loop header, condition check
_emit_block(body);     // loop body
_emit_block(update);   // increment
_emit_block(merge);    // after loop
```

`_loop_header_info` maps the XIR prepare block to `(merge_block, update_block)` so that `_emit_block` can emit `OpLoopMerge` at the loop header.

### Simple Loop (`_emit_simple_loop_inst`)

Like a `do-while` with body as header:
```
_loop_header_info[body] = (merge, body);
createBranch(body);
_emit_block(body);
_emit_block(merge);
```

### Branch / Conditional Branch

Direct `createBranch` or `createConditionalBranch`, then recursively `_emit_block` targets.

## Binding Generation (`bind.cpp`)

`generate_binding(Function kernel)` inspects kernel arguments and propagated builtin callables to determine descriptor layout, then creates SPIR-V global variables with `OpDecorate` bindings.

### Descriptor Layout (in order)

| Index | Type | SPIR-V Global | Notes |
|-------|------|---------------|-------|
| 0 | `ConstantValue` | PushConstant struct (`uint4`) | space=0, reg=0 |
| 1 | `SamplerHeap` | `UniformConstant` sampler array | space=1, reg=0, size=16 |
| 2 | `StructuredBuffer` (optional) | CBuffer for non-resource args | only if cbuffer non-empty |
| ... | `SRVBufferHeap` | bindless buffers | if `_use_buffer_bindless` |
| ... | `SRVTextureHeap` / `UAVTextureHeap` | bindless 2D/3D textures | if `_use_tex2d_bindless` / `_use_tex3d_bindless` |
| ... | Per-arg textures/buffers/accel | individual bindings | space=0, sequential registers |
| ... | `RWStructuredBuffer` x2 | print buffers | if `kernel.requires_printing()` |

### Property to SPIR-V Global Mapping

```cpp
for (auto &&prop : _properties) {
    switch (prop.type) {
        case ConstantValue:    // PushConstant struct with Block decoration
        case SamplerHeap:      // UniformConstant array, DescriptorSet + Binding
        case StructuredBuffer: // StorageBuffer struct with Block, RuntimeArray<uint>
        case RWStructuredBuffer:
        case SRVTextureHeap:   // UniformConstant image array (Dim2D, sampled=1)
        case UAVTextureHeap:   // UniformConstant image array (Dim2D, sampled=2)
        case SRVBufferHeap:    // StorageBuffer struct with Block
        case SPIRVAccel:       // UniformConstant AccelerationStructure
    }
    _property_ids.emplace_back(var);
}
```

Resource arguments are resolved via `_resolve_resource_argument()`, which computes an index into `_property_ids` based on the resource argument position plus the base offset (accounts for cbuffer, bindless flags).

## Kernel vs Callable Emission (`emit.cpp`)

### Kernel (`_emit_kernel`)

- Return type = void
- Parameters = only non-resource arguments
- Entry point: `addEntryPoint(GLCompute, func, "main")`
- Execution mode: `LocalSize(block_size.x, block_size.y, block_size.z)`
- Maps function param IDs to `xir::Argument` values

### Callable (`_emit_callable`)

- Return type = `_convert_type(callable->type())`
- Parameters = only non-resource arguments
- Name = `callable->name().value_or("callable")`
- No entry point / execution mode

## Block Emission (`_emit_block`)

```cpp
void SpirvCodegenEntry::_emit_block(const xir::BasicBlock *bb) {
    if (!bb || already emitted) return;
    auto spv_block = _get_or_create_block(bb);
    _builder.setBuildPoint(spv_block);
    if (bb is loop header) _builder.createLoopMerge(merge, continue_target, MaskNone);
    for (auto inst : bb->instructions()) _emit_instruction(inst);
}
```

## Adding a New Instruction

1. **If it's a new arithmetic op**: add case in `_emit_arithmetic_inst` (instruction.cpp), map to `spv::Op` or `GLSLstd450*` builtin.
2. **If it's a new resource op**: add case in `_emit_resource_read_inst`, `_emit_resource_write_inst`, or `_emit_resource_query_inst` (instruction.cpp).
3. **If it's a new thread-group op**: add case in `_emit_thread_group_inst` (instruction.cpp).
4. **If it's a new control-flow construct**: add method in `entry.h`, implement in `condition_inst.cpp`, dispatch from `_emit_instruction`.
5. **If it needs new binding support**: update `generate_binding()` in `bind.cpp` to create the SPIR-V global variable and decoration.
6. **If it introduces a new type**: update `_convert_type()` in `type.cpp`.

All new emission paths must:
- Use `_convert_type(inst->type())` for result type
- Use `_emit_value(operand)` for operands
- Store result in `_value_map.emplace(inst, id)` if `inst->type() != nullptr`
- Return `spv::NoResult` / not store for no-result instructions

## Precompiled Header (`lc_spirv_pch.h`)

Includes all standard Luisa headers (`luisa/core/*`, `luisa/ast/*`, `luisa/xir/*`, `luisa/runtime/*`) plus SPIRV headers (`SPIRV/SpvBuilder.h`, `SPIRV/disassemble.h`). Every `.cpp` in this folder includes `entry.h`, which transitively includes the PCH.

## Result Type

```cpp
struct SpirvResult {
    std::vector<uint32_t> spv_bin;
    vstd::vector<Property> properties;
    vstd::vector<std::pair<vstd::string, const Type *>> printers;
    bool useTex2DBindless;
    bool useTex3DBindless;
    bool useBufferBindless;
};
```

`properties` is consumed by the Vulkan backend descriptor set allocator. `printers` is used for host-side print formatting.
