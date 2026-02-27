# Ray Query Support for CUDA LLVM Codegen

## Overview

This document describes the implementation strategy for adding ray query support to the CUDA LLVM codegen path. The goal is to transform ray query loops from XIR into LLVM IR that's compatible with OptiX's ray tracing pipeline.

## Architecture

### Current Pipeline

```
XIR (RayQueryLoopInst, RayQueryDispatchInst)
    ↓
LLVM IR (with luisa.ray.query.* pseudo-intrinsics)
    ↓
RayQueryLoopExtraction Pass
    ↓
Extracted LLVM functions with pseudo-intrinsics
    ↓
Handler Splitting & Intrinsic Lowering Passes
    ↓
OptiX-compatible LLVM IR
    ↓
PTX Generation
    ↓
OptiX Pipeline (__raygen__main, __anyhit__ray_query, __intersection__ray_query)
```

### OptiX Program Structure

The generated PTX must contain these entry points:

1. **`__raygen__main`** - Kernel entry point that initiates ray queries
2. **`__anyhit__ray_query`** - Triangle intersection handler (dispatches to user code)
3. **`__intersection__ray_query`** - Procedural intersection handler (dispatches to user code)

The `__anyhit__` and `__intersection__` programs are defined in `cuda_device_resource.h` and dispatch to user-provided handlers based on query ID.

## Pseudo-Intrinsics

The LLVM codegen translates XIR ray query instructions into pseudo-intrinsics:

| Intrinsic | Purpose | Transformed To |
|-----------|---------|----------------|
| `luisa.ray.query.initialize` | Initialize ray query state | `luisa.ray.query.spawn` + context setup |
| `luisa.ray.query.spawn` | Start ray traversal | OptiX trace call with encoded context |
| `luisa.ray.query.proceed` | Continue to next candidate | `luisa.ray.query.dispatch` in extracted handler |
| `luisa.ray.query.dispatch` | Dispatch to next iteration | Loop back or return from handler |
| `luisa.ray.query.state` | Get current state (0=exit, 1=surface, 2=procedural) | Hit type decode |
| `luisa.ray.query.world.space.ray` | Get world space ray | `lc_ray_query_world_ray()` call |
| `luisa.ray.query.surface.candidate.hit` | Get triangle hit info | `lc_ray_query_triangle_candidate()` call |
| `luisa.ray.query.procedural.candidate.hit` | Get procedural hit info | `lc_ray_query_procedural_candidate()` call |
| `luisa.ray.query.committed.hit` | Get committed hit | `lc_ray_query_decode_hit()` call |
| `luisa.ray.query.commit.surface.hit` | Commit triangle hit | Set `result.committed = true` |
| `luisa.ray.query.commit.procedural.hit` | Commit procedural hit | Set `result.committed = true` + `result.t_hit = t` |
| `luisa.ray.query.terminate` | Terminate traversal | Set `result.terminated = true` |

## Implementation Phases

### Phase 1: Ray Query Loop Structure Analysis

**File**: `cuda_codegen_llvm_impl_rq_analysis.cpp` (new)

Identify and analyze ray query loops in LLVM IR:

```cpp
struct RayQueryLoopInfo {
    llvm::CallInst* initializeCall;      // luisa.ray.query.initialize
    llvm::CallInst* proceedCall;         // luisa.ray.query.proceed  
    llvm::Loop* loop;                    // LLVM loop containing proceed
    llvm::BasicBlock* dispatchBlock;     // Block with switch on state
    llvm::BasicBlock* surfaceBlock;      // Surface candidate handler
    llvm::BasicBlock* proceduralBlock;   // Procedural candidate handler
    llvm::BasicBlock* mergeBlock;        // Exit/merge block
    llvm::DenseSet<llvm::Value*> capturedInputs;   // Values used but not defined
    llvm::DenseSet<llvm::Value*> capturedOutputs;  // Values defined and used outside
};
```

**Key Analysis Steps**:

1. Find all `luisa.ray.query.proceed` calls
2. Use LLVM LoopInfo to find containing loop
3. Find dominating `luisa.ray.query.initialize` call
4. Identify dispatch block (block containing switch on `luisa.ray.query.state`)
5. Identify surface and procedural handler blocks from switch cases
6. Compute captured variables using liveness analysis

### Phase 2: Handler Extraction

**File**: `cuda_codegen_llvm_impl.cpp` - Modify `RayQueryLoopExtraction` pass

Instead of extracting the entire loop, extract individual handlers:

```cpp
// For each ray query:
// 1. Collect blocks reachable from surfaceBlock until dispatchBlock
auto surfaceBlocks = collectHandlerBlocks(surfaceBlock, dispatchBlock);

// 2. Collect blocks reachable from proceduralBlock until dispatchBlock  
auto proceduralBlocks = collectHandlerBlocks(proceduralBlock, dispatchBlock);

// 3. Extract surface handler
llvm::CodeExtractor surfaceExtractor{
    surfaceBlocks, &DT, false, nullptr, nullptr, AC
};
auto surfaceFunc = surfaceExtractor.extractCodeRegion(CEAC);
surfaceFunc->setName("lc_ray_query_triangle_intersection_0");
surfaceFunc->setLinkage(llvm::Function::InternalLinkage);

// 4. Extract procedural handler
llvm::CodeExtractor procExtractor{
    proceduralBlocks, &DT, false, nullptr, nullptr, AC
};
auto procFunc = procExtractor.extractCodeRegion(CEAC);
procFunc->setName("lc_ray_query_procedural_intersection_0");
procFunc->setLinkage(llvm::Function::InternalLinkage);

// 5. Replace original loop with trace call in kernel
```

### Phase 3: Handler Intrinsic Lowering

**File**: `cuda_codegen_llvm_impl_rq.cpp` - New function: `_lowerRayQueryHandler()`

Transform extracted handler functions to use device library calls:

```cpp
void _lowerRayQueryHandler(llvm::Function* handler, HandlerType type) {
    // Change signature to: LCIntersectionResult handler(void* ctx_in)
    auto voidPtrTy = llvm::PointerType::get(llvmContext, 0);
    auto resultTy = llvm::StructType::get(
        floatTy,   // t_hit
        i1Ty,      // committed
        i1Ty       // terminated
    );
    
    // Create new function with correct signature
    auto newFunc = llvm::Function::Create(
        llvm::FunctionType::get(resultTy, {voidPtrTy}, false),
        handler->getLinkage(),
        handler->getName(),
        handler->getParent()
    );
    
    // If no captured variables, ctx_in can be null
    // Insert at entry:
    // %ctx = bitcast void* %ctx_in to %ContextStruct*
    // %val = load from %ctx (only if non-null and needed)
    
    // Transform each instruction:
    for (auto& inst : instructions) {
        if (auto call = dyn_cast<CallInst>(&inst)) {
            auto callee = call->getCalledFunction();
            if (!callee) continue;
            
            auto name = callee->getName();
            if (name == "luisa.ray.query.world.space.ray") {
                // Replace with: call @lc_ray_query_world_ray()
                ReplaceWithDeviceCall(call, "lc_ray_query_world_ray");
            }
            else if (name == "luisa.ray.query.surface.candidate.hit") {
                // Replace with: call @lc_ray_query_triangle_candidate()
                ReplaceWithDeviceCall(call, "lc_ray_query_triangle_candidate");
            }
            else if (name == "luisa.ray.query.procedural.candidate.hit") {
                // Replace with: call @lc_ray_query_procedural_candidate()
                ReplaceWithDeviceCall(call, "lc_ray_query_procedural_candidate");
            }
            else if (name == "luisa.ray.query.commit.surface.hit") {
                // Store true to result.committed
                StoreToResultField(call, /*committed*/ true);
                call->eraseFromParent();
            }
            else if (name == "luisa.ray.query.commit.procedural.hit") {
                // Store true to result.committed and t to result.t_hit
                StoreProceduralCommit(call);
                call->eraseFromParent();
            }
            else if (name == "luisa.ray.query.terminate") {
                // Store true to result.terminated
                StoreToResultField(call, /*terminated*/ true);
                call->eraseFromParent();
            }
            else if (name == "luisa.ray.query.proceed") {
                // Replace with: call @luisa.ray.query.dispatch
                ReplaceIntrinsic(call, "luisa.ray.query.dispatch");
            }
        }
    }
    
    // Insert return of result struct at all exits
    InsertResultReturn(handler);
}
```

### Phase 4: Context Struct Generation

**File**: `cuda_codegen_llvm_impl_rq_context.cpp` (new)

Generate LLVM struct types for captured variables:

```cpp
llvm::StructType* createContextStruct(
    uint32_t queryId,
    const llvm::DenseSet<llvm::Value*>& capturedVars
) {
    if (capturedVars.empty()) {
        return nullptr;  // Use null pointer, no context needed
    }
    
    // Separate resources (pointers) from scalar values
    std::vector<llvm::Value*> resources;
    std::vector<llvm::Value*> scalars;
    
    for (auto* v : capturedVars) {
        if (v->getType()->isPointerTy()) {
            resources.push_back(v);
        } else {
            scalars.push_back(v);
        }
    }
    
    // Sort scalars by alignment (descending) to minimize padding
    std::sort(scalars.begin(), scalars.end(),
        [](auto a, auto b) {
            return getAlignment(a) > getAlignment(b);
        });
    
    // Build struct type
    llvm::SmallVector<llvm::Type*, 16> memberTypes;
    for (auto* v : resources) {
        memberTypes.push_back(v->getType());
    }
    for (auto* v : scalars) {
        memberTypes.push_back(v->getType());
    }
    
    auto structName = "RayQueryCtx" + std::to_string(queryId);
    return llvm::StructType::create(llvmContext, memberTypes, structName);
}
```

### Phase 5: Kernel-Side Transformation

**File**: `cuda_codegen_llvm_impl_func.cpp` - Modify kernel generation

Replace ray query loop in `__raygen__main`:

```cpp
// Before:
// loop:
//   call void @luisa.ray.query.proceed()
//   %state = call i8 @luisa.ray.query.state()
//   switch %state, label %exit [ ... ]

// After:
// if (!capturedVars.empty()) {
//   %ctx = alloca %RayQueryCtx0
//   ; store captured values to %ctx
//   %ctx_ptr = bitcast %RayQueryCtx0* %ctx to void*
// } else {
//   %ctx_ptr = null
// }
// call void @lc_ray_query_trace(%ray_query, i32 0, %ctx_ptr)
// if (!capturedVars.empty()) {
//   ; load updated captured values from %ctx
// }
// br %merge_block

void emitRayQueryTrace(
    llvm::IRBuilder<>& b,
    llvm::Value* rayQueryObj,
    uint32_t queryId,
    llvm::StructType* ctxStruct,
    const llvm::DenseSet<llvm::Value*>& capturedVars
) {
    llvm::Value* ctxPtr = nullptr;
    
    if (ctxStruct && !capturedVars.empty()) {
        // Allocate context on stack
        auto ctxAlloca = b.CreateAlloca(ctxStruct);
        ctxPtr = b.CreateBitCast(ctxAlloca, b.getPtrTy());
        
        // Store captured values
        uint32_t idx = 0;
        for (auto* v : capturedVars) {
            auto gep = b.CreateStructGEP(ctxStruct, ctxAlloca, idx++);
            b.CreateStore(v, gep);
        }
    } else {
        // No captured variables, use null
        ctxPtr = llvm::Constant::getNullValue(b.getPtrTy());
    }
    
    // Encode query ID in payload
    // r0 = (queryId << 24) | (ctx_ptr_high & 0xffffff)
    // r1 = ctx_ptr_low
    auto ctxInt = b.CreatePtrToInt(ctxPtr, b.getInt64Ty());
    auto ctxHigh = b.CreateLShr(ctxInt, 32);
    auto ctxHighMasked = b.CreateAnd(ctxHigh, 0xffffff);
    auto queryIdShifted = b.getInt32(queryId << 24);
    auto r0 = b.CreateOr(queryIdShifted, b.CreateTrunc(ctxHighMasked, b.getInt32Ty()));
    auto r1 = b.CreateTrunc(ctxInt, b.getInt32Ty());
    
    // Call lc_ray_query_trace
    auto traceFunc = module.getFunction("lc_ray_query_trace");
    b.CreateCall(traceFunc, {rayQueryObj, b.getInt32(queryId), ctxPtr});
    
    // If we had captured variables, load them back
    if (ctxStruct && !capturedVars.empty()) {
        uint32_t idx = 0;
        for (auto* v : capturedVars) {
            if (v is written to in handler) {
                auto gep = b.CreateStructGEP(ctxStruct, ctxAlloca, idx);
                auto loaded = b.CreateLoad(ctxStruct->getElementType(idx), gep);
                // Replace uses of v with loaded value after trace call
            }
            idx++;
        }
    }
}
```

### Phase 6: Pass Pipeline

**File**: `cuda_codegen_llvm_impl.cpp` - `generate()` function

Run passes in correct order:

```cpp
luisa::string CUDACodegenLLVMImpl::generate(const xir::Module& xir_module) noexcept {
    // Phase 1: Initial LLVM IR generation from XIR
    _analyze_ray_tracing_usage(xir_module);
    for (auto func : xir_module.function_list()) {
        if (auto def = func->definition()) {
            _translate_function(def);
        }
    }
    
    // Phase 2: Run initial optimization pass
    _run_optimization_passes();
    
    // Phase 3: Ray Query transformation passes
    llvm::ModulePassManager rqMPM;
    
    // 3a. Analyze ray query loops
    rqMPM.addPass(RayQueryAnalysisPass{});
    
    // 3b. Split and extract handlers
    rqMPM.addPass(RayQueryHandlerExtractionPass{});
    
    // 3c. Lower intrinsics in handlers
    rqMPM.addPass(RayQueryHandlerLoweringPass{});
    
    // 3d. Transform kernel-side loops
    rqMPM.addPass(RayQueryKernelTransformPass{});
    
    llvm::ModuleAnalysisManager rqMAM;
    rqMAM.registerPass([] { return llvm::LoopAnalysis(); });
    rqMAM.registerPass([] { return llvm::DominatorTreeAnalysis(); });
    rqMAM.registerPass([] { return llvm::AssumptionAnalysis(); });
    // ... other analyses
    
    rqMPM.run(*_llvm_module, rqMAM);
    
    // Phase 4: Final optimization pass
    _run_optimization_passes();
    
    // Phase 5: Generate PTX
    return _generate_ptx();
}
```

## Critical Implementation Details

### Inline Assembly with Side Effects

When calling OptiX intrinsics, always mark with side effects:

```cpp
llvm::InlineAsm* getOptixInlineAsm(
    const char* asmString,
    const char* constraints,
    bool hasSideEffects
) {
    return llvm::InlineAsm::get(
        llvm::FunctionType::get(retTy, argTypes, false),
        asmString,
        constraints,
        hasSideEffects,  // MUST be true for OptiX calls
        false,           // isAlignStack
        llvm::InlineAsm::AsmDialect::AD_ATT
    );
}

// Example: OptiX report intersection
auto asmStr = "call ($0), _optix_report_intersection_0, ($1, $2);";
auto constraints = "=r,f,r";
auto asmFunc = getOptixInlineAsm(asmStr, constraints, true);
b.CreateCall(asmFunc, {hitKind, tHit});
```

### Function Attributes

Mark entry points correctly:

```cpp
// __raygen__main
llvmKernel->setCallingConv(llvm::CallingConv::PTX_Kernel);
llvmKernel->setLinkage(llvm::Function::ExternalLinkage);

// __anyhit__ray_query and __intersection__ray_query
// These are defined in cuda_device_resource.h and linked in

// Extracted handlers (internal linkage)
surfaceFunc->setLinkage(llvm::Function::InternalLinkage);
surfaceFunc->addFnAttr(llvm::Attribute::AlwaysInline);
```

### Null Context Handling

When there are no captured variables:

```cpp
// Pass null pointer - handlers should check before loading
if (ctx_in != nullptr) {
    auto ctx = *static_cast<ContextStruct*>(ctx_in);
    // ... use ctx
} else {
    // No captured variables, use defaults
}
```

Handlers must guard all context loads with null checks.

### State Machine Optimization

The switch on `luisa.ray.query.state()` should be constant-folded:

```cpp
// In surface handler (constant propagated):
%state = call i8 @luisa.ray.query.state()
; LLVM knows this always returns 1 (surface_candidate)
; Switch becomes unconditional branch to surface code
; Dead procedural branch is eliminated
```

Run constant folding and simplify CFG passes after handler extraction.

## Context Encoding

Query ID and context pointer are encoded in OptiX payload registers:

```
Payload Register 0 (r0):
  Bits [31:24] - Query ID (0-31)
  Bits [23:0]  - High 24 bits of context pointer

Payload Register 1 (r1):
  Bits [31:0]  - Low 32 bits of context pointer
```

Encoding:
```cpp
auto ctx_u64 = reinterpret_cast<uint64_t>(ctx_ptr);
auto r0 = (query_id << 24) | ((ctx_u64 >> 32) & 0xffffff);
auto r1 = ctx_u64 & 0xffffffff;
```

Decoding (in device library):
```cpp
auto query_id = (payload0 >> 24) & 0xff;
auto ctx_ptr = ((payload0 & 0xffffff) << 32) | payload1;
```

## Testing Strategy

### Test 1: Basic Triangle Query
```cpp
Kernel2D kernel = [&]() {
    Var<Ray> ray = make_ray(origin, direction);
    Var<CommittedHit> hit = accel->traverse(ray, {})
        .on_surface_candidate([&](SurfaceCandidate& c) {
            c.commit();
        })
        .trace();
};
```

### Test 2: Basic Procedural Query
```cpp
Kernel2D kernel = [&]() {
    Var<Ray> ray = make_ray(origin, direction);
    Var<CommittedHit> hit = accel->traverse(ray, {})
        .on_procedural_candidate([&](ProceduralCandidate& c) {
            c.commit(1.0f);
        })
        .trace();
};
```

### Test 3: Captured Variables
```cpp
Kernel2D kernel = [&]() {
    Float3 color = make_float3(1, 0, 0);
    Var<CommittedHit> hit = accel->traverse(ray, {})
        .on_surface_candidate([&](SurfaceCandidate& c) {
            color = make_float3(0, 1, 0);
            c.commit();
        })
        .trace();
    // color should be (0, 1, 0) after trace
};
```

### Test 4: Resource Access
```cpp
Kernel2D kernel = [&]() {
    Buffer<float> buf = ...;
    Var<CommittedHit> hit = accel->traverse(ray, {})
        .on_surface_candidate([&](SurfaceCandidate& c) {
            float val = buf->read(0);
            if (val > 0.5f) c.commit();
        })
        .trace();
};
```

### Test 5: Multiple Ray Queries
```cpp
Kernel2D kernel = [&]() {
    // First ray query
    auto hit1 = accel->traverse(ray1, {}).on_surface_candidate(...).trace();
    
    // Second ray query  
    auto hit2 = accel->traverse(ray2, {}).on_procedural_candidate(...).trace();
};
```

### Test 6: Complex Control Flow
```cpp
Kernel2D kernel = [&]() {
    Float t = 0.0f;
    Var<CommittedHit> hit = accel->traverse(ray, {})
        .on_surface_candidate([&](SurfaceCandidate& c) {
            $if (c.hit().t > 0.5f) {
                t = c.hit().t;
                c.commit();
            };
        })
        .trace();
};
```

### Comparison Testing

For each test, compare:
1. Output image with AST codegen
2. Output image with LLVM codegen  
3. Pixel values should match within floating-point tolerance

## Files to Modify

| File | Changes |
|------|---------|
| `cuda_codegen_llvm_impl.cpp` | Modify `RayQueryLoopExtraction` pass, add pass pipeline |
| `cuda_codegen_llvm_impl_rq.cpp` | Add handler lowering functions |
| `cuda_codegen_llvm_impl_rq_analysis.cpp` | New: Ray query analysis |
| `cuda_codegen_llvm_impl_rq_context.cpp` | New: Context struct generation |
| `cuda_codegen_llvm_impl_func.cpp` | Modify kernel generation for ray queries |
| `cuda_codegen_llvm_impl_resource.cpp` | Add OptiX inline ASM helpers |

## Open Questions (Resolved)

1. **Handler inlining**: Handlers should be inlined (mark with `AlwaysInline`). Entry points (`__raygen__`, `__anyhit__`, `__intersection__`) are marked as `PTX_Kernel` with external linkage to prevent inlining.

2. **Multiple ray queries**: Use switch-case dispatch based on unique query ID (0-31).

3. **Empty context**: Use null pointer for empty context. Handlers must guard all loads with null checks.

4. **Optimization passes**: Run initial opt pass, then RQ transforms, then final opt pass.

## References

- `cuda_device_resource.h` - OptiX shader entry points and handler dispatch
- `cuda_codegen_ast.cpp` - AST-based ray query lowering (reference implementation)
- `lower_ray_query_loop.cpp` - XIR ray query loop lowering (understanding structure)
- `fallback_codegen.cpp` - CPU backend handler extraction (reference patterns)

## Notes

- Maximum 32 ray queries per kernel (limited by query ID encoding)
- Handlers must not call other ray queries (no recursion)
- Context struct alignment is critical for correctness
- Always use `hasSideEffects=true` for OptiX inline ASM
- Opaque pointers (LLVM 15+) - use `b.getPtrTy()` for pointer types
