# Ray Query Lowering in CUDA LLVM Backend

This document describes how inline ray query loops are lowered into the OptiX ray tracing pipeline using the LLVM backend.

## Current Status (Feb 27, 2026)

**Implementation Status: ✅ COMPLETE AND VERIFIED**

The ray query lowering implementation is now fully functional and all tests pass. The implementation correctly transforms inline ray query loops into OptiX-compatible intersection programs.

### What Works:

1. ✅ **Loop Extraction Pass**: The `RayQueryLoopExtraction` LLVM pass successfully identifies and extracts ray query loops into functions named `ray.query.loop.extracted`.
2. ✅ **Ray Query Intrinsics**: All ray query intrinsics are properly emitted and handled:
   - `luisa.ray.query.initialize` - Spawn call for OptiX trace
   - `luisa.ray.query.proceed` - State dispatch  
   - `luisa.ray.query.state` - Returns hit type (surface/procedural)
   - `luisa.ray.query.commit.*` - Commit hits
   - `luisa.ray.query.terminate` - Terminate query
3. ✅ **Ray Query Object Read/Write**: XIR instructions properly translated to placeholder intrinsics in `cuda_codegen_llvm_impl_rq.cpp`, then lowered to OptiX calls during materialization
4. ✅ **Materialization Pipeline**: The `_materialize_ray_query_loops()` function successfully creates intersection programs
5. ✅ **Manual Function Cloning**: Custom cloning implementation that properly handles PHI nodes and address spaces

### Verified Tests:
- ✅ `test_procedural` - Complex ray query with surface and procedural candidates
- ✅ `test_procedural_callable` - Ray queries with callable support
- ✅ `test_rq_simple` - Basic ray query functionality
- ✅ `test_path_tracing_cutout` - Path tracing with ray queries

## Overview

LuisaCompute's Ray Query allows for custom intersection logic within a kernel using a loop-like structure. In the CUDA backend, these queries are implemented using OptiX's `optixTrace` and specialized hit groups.

The lowering process consists of several stages:

### Stage 1: Loop Extraction

The `RayQueryLoopExtraction` LLVM module pass (in `cuda_codegen_llvm_impl.cpp`) identifies ray query loops and extracts them:

1. **Identify Loops**: Find loops containing `luisa.ray.query.proceed` calls
2. **Find Initialize**: Locate the `luisa.ray.query.initialize` call that dominates the loop
3. **Demote PHI Nodes**: Convert PHI nodes in the loop header to stack allocas to preserve state
4. **Extract Function**: Use LLVM's `CodeExtractor` to pull the loop body into a new function named `ray.query.loop.extracted`
5. **Rename Intrinsics**: 
   - `initialize` → `spawn` (the trace call)
   - `proceed` → `dispatch` (the switch dispatch)

The extracted function contains:
- The dispatch switch with candidate handler blocks
- Ray query intrinsics (state, commit, terminate)
- User-provided candidate handler code

### Stage 2: Materialization

The `_materialize_ray_query_loops()` function (in `cuda_codegen_llvm_impl_rq.cpp`) transforms the extracted function into OptiX-compatible intersection programs:

#### 2.1 Find Loop Call Site

For each extracted function `F`:
1. Find the call instruction (`loop_call`) that invokes `F`
2. **Find Spawn Call**: Search the caller function (not `F`) for the `luisa.ray.query.spawn` intrinsic call
3. Extract spawn arguments: accel, ray, time, mask, flags

**Important**: The spawn call remains in the caller function after extraction, not inside the extracted loop function.

#### 2.2 Create Intersection Functions

Generate two specialized intersection functions:
- `lc_ray_query_triangle_intersection_i`: Handles surface candidates
- `lc_ray_query_procedural_intersection_i`: Handles procedural candidates

Each function:
1. **Clone the Extracted Function**: Manual cloning to handle PHI nodes:
   - Create corresponding basic blocks
   - Clone instructions using `llvm::RemapInstruction`
   - Add block mappings to the value map for PHI node handling
   - Replace argument uses

2. **Allocate Result Variables**:
   ```cpp
   float t_hit = 0.0;
   bool committed = false;
   bool terminated = false;
   ```

3. **Lower Intrinsics**:
   - `state` → Constant value (1 for triangle, 2 for procedural)
   - `commit_surface_hit` → `committed = true`
   - `commit_procedural_hit(t)` → `committed = true; t_hit = t`
   - `terminate` → `terminated = true`
   - `dispatch` → Removed (control flow simplified)

4. **Transform Control Flow**:
   - Replace the switch with direct branches to the appropriate candidate handler
   - Redirect backedges (branches to dispatch) to a return block
   - Redirect exit branches to return with `terminated = true`
   - Return block packs `{t_hit, committed, terminated}` into result struct

#### 2.3 Generate OptiX Entry Points

Two global entry point functions are generated:

**`__anyhit__ray_query`** (for triangle intersections):
1. Extract implementation tag and context pointer from payload registers
2. Switch on implementation tag to call the correct `lc_ray_query_triangle_intersection_i`
3. If `committed`: Call `_optix_hitobject_make_hit`
4. If `terminated`: Call `_optix_terminate_ray`
5. Otherwise: Call `_optix_ignore_intersection`

**`__intersection__ray_query`** (for procedural intersections):
1. Extract implementation tag and context pointer from payload registers  
2. Switch on implementation tag to call the correct `lc_ray_query_procedural_intersection_i`
3. If `committed`: Call `_optix_report_intersection` with `t_hit` and hit kind

### Stage 3: Lower Ray Query Intrinsics

After materialization, the `_lower_ray_query_intrinsics()` function replaces placeholder intrinsics with actual OptiX calls:

- `world.space.ray` → `_optix_get_world_ray_*`
- `surface.candidate.hit` → `_optix_read_instance_idx`, `_optix_read_primitive_idx`, `_optix_get_triangle_barycentrics`, `_optix_get_ray_tmax`
- `procedural.candidate.hit` → `_optix_read_instance_idx`, `_optix_read_primitive_idx`
- `committed.hit` → `_optix_hitobject_is_hit`, `_optix_hitobject_get_instance_idx`, etc.

## Critical Implementation Details

### Spawn Call Location

**Bug Fixed**: The materialization code originally searched for the spawn call inside the extracted function, but it's actually in the caller function. The fix searches all basic blocks in the caller function (`loop_call->getFunction()`).

### Side Effects Preservation

**Bug Fixed**: LLVM was optimizing away the `initialize`/`spawn` call because it returns void and appeared to have no side effects. The fix adds a `sideeffect` parameter to `_call_ray_query_intrinsic()` that removes `ReadNone` and `ReadOnly` attributes from the function.

### Placeholder Intrinsics

**Bug Fixed**: Direct OptiX intrinsics (like `_optix_read_instance_idx`) were being called during initial code generation, but they're only valid in intersection/any-hit contexts. The fix uses placeholder intrinsics (e.g., `luisa.ray.query.surface.candidate.hit`) that get lowered to OptiX calls during materialization.

### Address Space Handling

The intersection functions use address space 1 (CUDA global memory) for pointers, matching the original extracted function. This is critical for correct memory access in the OptiX pipeline.

### Payload Type ID

The payload type ID must be `2` (`LC_PAYLOAD_TYPE_ID_1`) for ray queries, matching the AST backend convention.

## Payload Layout

The Ray Query pipeline uses 2 custom payload registers:
- `r0`: `(impl_tag << 24) | (p_ctx_hi & 0xffffff)`
- `r1`: `p_ctx_lo`

The `impl_tag` is used to index into the switch-case in the entry points to find the correct intersection logic. The context pointer (`p_ctx`) points to the capture struct containing all variables captured by the ray query loop.

Packing format (matching AST backend):
```cpp
r0 = (impl_tag << 24) | (static_cast<uint32_t>(ctx_ptr >> 32) & 0xffffff);
r1 = static_cast<uint32_t>(ctx_ptr);
```

## Runtime Configuration

To use the LLVM codegen for CUDA:
1. Enable it in CMake: `-DLUISA_COMPUTE_ENABLE_EXPERIMENTAL_CUDA_LLVM_CODEGEN=ON`
2. Enable it at runtime: Set the environment variable `LUISA_EXPERIMENTAL_LLVM_CODEGEN=1`

## Related Files

- `src/backends/cuda/llvm_codegen/cuda_codegen_llvm_impl_rq.cpp` - Main ray query lowering implementation
  - `_call_ray_query_intrinsic()` - Creates ray query intrinsic calls with optional side effects
  - `_translate_ray_query_object_read_inst()` - Translates XIR read instructions to placeholder intrinsics
  - `_translate_ray_query_object_write_inst()` - Translates XIR write instructions to placeholder intrinsics
  - `_lower_ray_query_intrinsics()` - Lowers placeholder intrinsics to OptiX calls
  - `_materialize_ray_query_loops()` - Main materialization pipeline

- `src/backends/cuda/llvm_codegen/cuda_codegen_llvm_impl.cpp` - RayQueryLoopExtraction pass
  - `RayQueryLoopExtraction::extractRayQueryLoops()` - Identifies and extracts ray query loops

- `src/backends/cuda/llvm_codegen/cuda_codegen_llvm_impl_resource.cpp` - Ray query initialization intrinsic
  - Emits `luisa.ray.query.initialize` with sideeffect flag

- `src/backends/cuda/llvm_codegen/cuda_codegen_llvm_impl.h` - Intrinsics names and helper declarations

## Testing

Run tests with LLVM codegen:
```bash
LUISA_EXPERIMENTAL_LLVM_CODEGEN=1 ./bin/test_procedural cuda
LUISA_EXPERIMENTAL_LLVM_CODEGEN=1 ./bin/test_procedural_callable cuda
LUISA_EXPERIMENTAL_LLVM_CODEGEN=1 ./bin/test_rq_simple cuda
LUISA_EXPERIMENTAL_LLVM_CODEGEN=1 ./bin/test_path_tracing_cutout cuda
```

All tests should pass and produce identical output to the AST backend.
