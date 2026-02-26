# Ray Query Lowering in CUDA LLVM Backend

This document describes how inline ray query loops are lowered into the OptiX ray tracing pipeline using the LLVM backend.

## Current Status (Feb 26, 2026)

**Implementation Status: ✅ COMPLETE**

The ray query lowering implementation is now fully functional. The critical cloning issue has been resolved by implementing a custom manual cloning strategy that properly handles PHI nodes and address spaces.

### What Works:

1. ✅ **Loop Extraction Pass**: The `RayQueryLoopExtraction` LLVM pass successfully identifies and extracts ray query loops into functions named `ray.query.loop.extracted`.
2. ✅ **Ray Query Intrinsics**: All ray query intrinsics are properly emitted and handled:
   - `luisa.ray.query.initialize` - Spawn call for OptiX trace
   - `luisa.ray.query.proceed` - State dispatch
   - `luisa.ray.query.state` - Returns hit type (surface/procedural)
   - `luisa.ray.query.commit.*` - Commit hits
   - `luisa.ray.query.terminate` - Terminate query
3. ✅ **Ray Query Object Read/Write**: XIR instructions properly translated to LLVM intrinsics in `cuda_codegen_llvm_impl_rq.cpp`
4. ✅ **Materialization Pipeline**: The `_materialize_ray_query_loops()` function successfully creates intersection programs
5. ✅ **Manual Function Cloning**: Custom cloning implementation that properly handles:
   - PHI node operands and incoming blocks
   - Address space preservation (addrspace 1 for CUDA global memory)
   - Argument remapping between original and cloned functions

### Implementation Details:

**Function Cloning Strategy** (lines 221-315 in `cuda_codegen_llvm_impl_rq.cpp`):

Instead of using `CloneFunctionInto` (which hangs due to PHI node cycles) or `CloneFunction` (which has blockaddress issues), we implemented a custom manual cloning approach:

1. Create corresponding basic blocks in the new function
2. Clone instructions one by one using `llvm::RemapInstruction`
3. Add block mappings to the value map so PHI nodes get proper block references
4. Manually replace uses of the old function's argument with the new function's argument
5. Handle return instructions specially (new function has different return type)

**Key Fixes Applied**:

1. **Address Space Handling**: The intersection functions and entry points now use the correct pointer address space (addrspace 1) matching the original extracted function
2. **Payload Type ID**: Fixed incorrect payload type ID (was 5, should be 2 = `LC_PAYLOAD_TYPE_ID_1`)
3. **PHI Node Processing**: Used `llvm::RemapInstruction` with block mappings to properly handle PHI node operands
4. **Argument Replacement**: Implemented fallback iteration to find and replace all argument uses when vmap lookup fails

## Overview

LuisaCompute's Ray Query allows for custom intersection logic within a kernel using a loop-like structure. In the CUDA backend, these queries are implemented using OptiX's `optixTrace` and specialized hit groups.

The lowering process consists of several stages:

1.  **Loop Extraction:** Outermost ray query loops are identified and their bodies are extracted into separate functions named `ray.query.loop.extracted`.
2.  **Intersection Program Generation:** Each extracted function is cloned into two specialized intersection programs:
    *   `lc_ray_query_triangle_intersection_i`: Handles triangle candidate hits.
    *   `lc_ray_query_procedural_intersection_i`: Handles procedural candidate hits.
3.  **Intrinsic Lowering:** `luisa.ray.query.*` intrinsics within these programs (and the main kernel) are replaced with OptiX-specific calls:
    *   `world.space.ray` -> `_optix_get_world_ray_*`
    *   `surface.candidate.hit` / `procedural.candidate.hit` -> `_optix_read_instance_idx`, `_optix_read_primitive_idx`, `_optix_get_triangle_barycentrics`, etc.
    *   `committed.hit` -> `_optix_hitobject_is_hit`, `_optix_hitobject_get_instance_idx`, etc.
    *   `commit.*` -> Sets a local `committed` flag and optionally stores `t_hit`.
    *   `terminate` -> Sets a local `terminated` flag.
4.  **Control Flow Transformation:** The extracted functions are modified to return an `LCIntersectionResult` structure:
    ```cpp
    struct LCIntersectionResult {
        float t_hit;
        bool committed;
        bool terminated;
    };
    ```
    Loop backedges are replaced with branches to a return block that packs these flags.
5.  **OptiX Entry Points:** Two global entry points are generated:
    *   `__anyhit__ray_query`: Dispatches to the correct `lc_ray_query_triangle_intersection_i` based on the implementation tag in the payload. It calls `_optix_ignore_intersection` if not committed and `_optix_terminate_ray` if terminated.
    *   `__intersection__ray_query`: Dispatches to the correct `lc_ray_query_procedural_intersection_i`. It calls `_optix_report_intersection` with the appropriate hit kind and `t_hit` if committed.

## Payload Layout

The Ray Query pipeline uses 2 custom payload registers:
*   `r0`: `(impl_tag << 24) | (p_ctx_hi & 0xffffff)`
*   `r1`: `p_ctx_lo`

The `impl_tag` is used to index into the switch-case in the entry points to find the correct intersection logic. The context pointer (`p_ctx`) points to the capture struct containing all variables captured by the ray query loop.

## Runtime Configuration

To use the LLVM codegen for CUDA:
1.  Enable it in CMake: `-DLUISA_COMPUTE_ENABLE_EXPERIMENTAL_CUDA_LLVM_CODEGEN=ON`.
2.  Enable it at runtime: Set the environment variable `LUISA_EXPERIMENTAL_LLVM_CODEGEN=1`.

## Related Files

- `src/backends/cuda/llvm_codegen/cuda_codegen_llvm_impl_rq.cpp` - Main ray query lowering implementation (lines 127-395)
- `src/backends/cuda/llvm_codegen/cuda_codegen_llvm_impl.cpp` - RayQueryLoopExtraction pass (lines 279-376)
- `src/backends/cuda/llvm_codegen/cuda_codegen_llvm_impl_resource.cpp` - Ray query initialization intrinsic
- `src/backends/cuda/llvm_codegen/cuda_codegen_llvm_impl.h` - Intrinsics names and helper declarations
