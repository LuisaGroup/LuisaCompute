# Ray Query Support for CUDA LLVM Codegen

## Status: ✅ IMPLEMENTED

This document describes the implementation of ray query support in the CUDA LLVM codegen path. The implementation transforms XIR ray query instructions into LLVM IR that's compatible with OptiX's ray tracing pipeline.

## Test Results

| Test | Status | Notes |
|------|--------|-------|
| `test_ray_query_simple` | ✅ **PASS** | Both Test 1 & Test 2 working |
| `test_procedural` | ✅ **PASS** | Procedural intersection working |
| `test_procedural_callable` | ✅ **PASS** | Callable programs with ray queries |

## Architecture

### Pipeline Overview

```
XIR (RayQueryLoopInst, RayQueryDispatchInst)
    ↓
LLVM IR (with luisa.ray.query.* pseudo-intrinsics)
    ↓
RayQueryLoopExtraction Pass (extracts loop into handler function)
    ↓
Remove handler call from kernel (CRITICAL - prevents illegal inlining)
    ↓
Handler Intrinsic Lowering (transform to OptiX device calls)
    ↓
OptiX Entry Point Generation (__anyhit__, __intersection__)
    ↓
Inline Pass (only into entry points, not kernel)
    ↓
PTX Generation
    ↓
OptiX Pipeline
```

### Key Components

1. **`__raygen__main`** - Kernel entry point, initiates ray queries via spawn call
2. **`__anyhit__ray_query`** - Triangle intersection handler (calls all handlers)
3. **`__intersection__ray_query`** - Procedural intersection handler (calls procedural-only handlers)

## Critical Implementation Lessons

### 1. Handler Detection via Intrinsic Scanning

**NEVER** rely on fragile name matching or stored pointers. Use dynamic scanning:

```cpp
[[nodiscard]] bool _is_ray_query_handler(llvm::Function *func) noexcept {
    if (func == nullptr || func->isDeclaration()) return false;
    
    // Check for dispatch intrinsic OR ray.query.dispatch block
    for (auto &block : *func) {
        if (block.getName().contains("ray.query.dispatch")) {
            return true;
        }
        for (auto &inst : block) {
            if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
                auto *callee = call->getCalledFunction();
                if (callee && callee->getName() == "luisa.ray.query.dispatch") {
                    return true;
                }
            }
        }
    }
    return false;
}
```

**Why this matters**: Functions can be deleted by dead code elimination, merged by optimization passes, or renamed. Scanning always finds the current state.

### 2. Remove Handler Call from Kernel (CRITICAL)

After extracting the ray query loop, **you MUST remove the call to the handler from the kernel**:

```cpp
// In RayQueryLoopExtraction pass, after extractCodeRegion:
for (auto &BB : *F) {
    for (auto It = BB.begin(); It != BB.end();) {
        auto *CI = llvm::dyn_cast<llvm::CallInst>(&*It++);
        if (CI && CI->getCalledFunction() == NewF) {
            CI->eraseFromParent();  // REMOVE THE CALL
        }
    }
}
```

**Why this matters**: OptiX will inline the handler into ANY function that calls it. If the kernel calls the handler, OptiX inlines the handler code (with `optixGetTriangleBarycentrics` calls) into `__raygen__main`, which is **ILLEGAL** and causes "Illegal call to optixGetTriangleBarycentrics in function __raygen__main".

### 3. Inline Pass Must Be Selective

Only inline handlers into OptiX entry points, not the kernel:

```cpp
for (auto &func : *_llvm_module) {
    auto func_name = func.getName();
    bool is_optix_entry = (func_name == "__anyhit__ray_query" || 
                           func_name == "__intersection__ray_query");
    if (!is_optix_entry) continue;
    
    // Only inline handlers in entry points...
}
```

### 4. Disable MergeFunctions for Ray Query Shaders

MergeFunctions will merge `__anyhit__` and `__intersection__` programs if they look similar:

```cpp
PTO.MergeFunctions = !_rt_analysis.uses_ray_query;
```

### 5. Surface vs Procedural Handler Filtering

Handlers that read triangle barycentrics (`optixGetTriangleBarycentrics`) are **ONLY** valid in `__anyhit__` programs, not `__intersection__`:

```cpp
// Check BEFORE lowering intrinsics
bool has_surface = _handler_has_surface_candidate_hit(func);

// In intersection program, skip surface handlers:
if (!handler_has_surface_hit[i]) {
    procedural_handlers.push_back(handlers[i]);
}
```

### 6. Filter Non-Void Handlers

Some extracted functions return values (e.g., `i1`). These are NOT valid ray query handlers for OptiX:

```cpp
if (!handler->getReturnType()->isVoidTy()) {
    // Skip this handler - not a valid ray query handler
}
```

### 7. Handler Argument Passing

Handlers may have multiple pointer arguments. Pass null for buffer pointers, ctx_ptr for output params:

```cpp
llvm::SmallVector<llvm::Value *, 4> handler_args;
for (auto &arg : handler->args()) {
    auto arg_type = arg.getType();
    if (arg_type->isPointerTy()) {
        auto ptr_type = llvm::dyn_cast<llvm::PointerType>(arg_type);
        if (ptr_type && ptr_type->getAddressSpace() == 0) {
            handler_args.push_back(ctx_ptr);  // Context pointer
        } else {
            handler_args.push_back(llvm::ConstantPointerNull::get(ptr_type));
        }
    }
}
```

### 8. Use NoDuplicate and NoInline on Entry Points

Prevent optimization passes from merging entry points:

```cpp
func->addFnAttr(llvm::Attribute::NoDuplicate);
func->addFnAttr(llvm::Attribute::NoInline);
```

## Pseudo-Intrinsics

| Intrinsic | Implementation |
|-----------|----------------|
| `luisa.ray.query.spawn` | OptiX trace call with encoded context |
| `luisa.ray.query.dispatch` | No-op (removed) |
| `luisa.ray.query.state` | Returns constant based on handler type |
| `luisa.ray.query.world.space.ray` | `_optix_get_world_ray` inline asm |
| `luisa.ray.query.surface.candidate.hit` | `_optix_read_instance_index`, `_optix_read_primitive_index`, `_optix_get_triangle_barycentrics` |
| `luisa.ray.query.procedural.candidate.hit` | `_optix_read_instance_index`, `_optix_read_primitive_index` |
| `luisa.ray.query.commit.surface.hit` | Set `result.committed = true` |
| `luisa.ray.query.commit.procedural.hit` | Set `result.committed = true` |
| `luisa.ray.query.terminate` | Set `result.terminated = true` |

## LLVM Intrinsics in Handlers

Handlers may contain LLVM intrinsics that need to be lowered:

- `llvm.vector.reduce.fadd.vNf32` → Extract elements and sum
- `llvm.nvvm.sqrt.approx.ftz.f` → `llvm::Intrinsic::sqrt`
- `llvm.nvvm.rsqrt.approx.ftz.f` → `1.0 / sqrt(x)`

## Result Struct

```cpp
// LCIntersectionResult: { i8 committed, i8 terminated }
struct_type = llvm::StructType::get(
    llvm::Type::getInt8Ty(context),   // committed
    llvm::Type::getInt8Ty(context)    // terminated
);
```

## Entry Point Structure

### __anyhit__ray_query

```llvm
define ptx_kernel void @__anyhit__ray_query() {
entry:
  call @optix_set_payload_types(2)
  %query_id = call @optix_get_payload(0)
  %ctx_ptr = call @optix_get_payload(1)
  switch %query_id, label %default [
    i32 0, label %handler_0
    i32 1, label %handler_1
  ]

handler_0:
  call void @ray.query.handler.0(%ctx_ptr)
  ret void

default:
  ret void
}
```

### __intersection__ray_query

Same structure but only dispatches to procedural handlers (those without `surface.candidate.hit`).

## Context Encoding

```
r0 = (query_id << 24) | (ctx_ptr_high & 0xffffff)
r1 = ctx_ptr_low
```

## Files Modified

- `cuda_codegen_llvm_impl.cpp` - Main implementation
- `cuda_codegen_llvm_impl.h` - Declarations and constants
- `cuda_codegen_llvm_impl_type.cpp` - Type definitions
- `cuda_codegen_llvm_impl_resource.cpp` - OptiX inline ASM helpers

## Key Constants

```cpp
static constexpr uint32_t llvm_payload_type_ray_query = 2;
static constexpr uint32_t llvm_hit_type_miss = 0;
static constexpr uint32_t llvm_hit_type_builtin = 1;
static constexpr uint32_t llvm_hit_type_procedural = 2;
static constexpr uint32_t llvm_hit_kind_procedural = 0x01;
static constexpr uint32_t llvm_hit_kind_procedural_terminated = 0x02;
```

## Lessons Learned

1. **Never use Function::mutateType()** - It's dangerous and doesn't work
2. **Always use intrinsic scanning** - Name matching breaks with optimization
3. **Remove handler calls from kernel** - Critical for preventing illegal OptiX inlining
4. **Test constantly** - Run tests after every change to catch issues early
5. **Filter handlers carefully** - Surface handlers can't go in intersection programs
6. **Handle all LLVM intrinsics** - Vector reductions, math intrinsics, etc.

## Future Improvements

- Support for context struct with captured variables
- More efficient payload encoding
- Support for ray query continuation
- Better handling of complex control flow

## References

- `cuda_device_resource.h` - OptiX inline ASM reference
- `cuda_codegen_xir.cpp` - AST codegen reference implementation
- OptiX Programming Guide - Entry point semantics
