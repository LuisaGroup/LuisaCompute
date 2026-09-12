# Tile XIR→AST Fallback (DX / VK)

`DeviceInterface::create_tile_kernel` is an optional native Tile compiler hook.
CUDA, Metal, Metal4, and SIMD/CPU implement it natively; DX and VK do not have a
native Tile compiler, so they realize Tile kernels through a shared fallback in
`src/backends/common/tile_xir_kernel.h`:

```
TileIR --(tile::bridge::xir::plan/lower)--> XIR SSA/CFG --(xir2ast)--> AST
      --(DeviceInterface::create_shader)--> HLSL/DXIL (DX) or SPIR-V (VK)
```

## Why this pipeline

- The TileIR→XIR bridge (`include/luisa/tile/bridge/xir/`) produces an
  in-memory, verified SSA/CFG module directly from TileIR — no AST or TVM
  intermediate. It is always built (only the TIRx bridge is optional).
- The XIR→AST translator runs the shared normalize pipeline (CFG restructure,
  reg2mem, no-PHI guarantee) and preserves the kernel block size, so the
  ordinary `create_shader` entry can compile the result with its standard
  buffer-argument ABI. This is the same XIR→AST→`create_shader` round-trip the
  DX/VK backends already prove with autodiff (`xir_autodiff.h`).
- The runtime and `tile::compile()` stay untouched: the runtime owns the
  shader/dispatch ABI and does not depend on the Tile compiler. Each backend
  adds one adapter translation unit (`LCDeviceTile.cpp`, `device_tile.cpp`)
  that supplies device facts (warp size, max block width, local snapshot
  budget) and delegates to `backend_detail::create_tile_kernel_via_ast`.

## Fail-closed option matrix

Every rejection is reported through `KernelMetadata::error`; no exception
escapes and no partial realization is returned:

| Request | Result |
|---|---|
| `ShaderOption::compile_only` | rejected (archives unsupported) |
| `CompileOptions::lowering == Lowering::TIRX` / `tirx != nullptr` | rejected (TIRx is not used on DX/VK; it carries a TVM dependency and only Metal/CUDA/PTX artifact formats) |
| `PlannerOptions::blocks_per_task` / `search_task_grain` | rejected (GPU target declares `supports_task_grain() == false`) |
| `PlannerOptions::local_lanes` = packet width | rejected (v1 disables local distribution; see below) |
| block width not a multiple of the warp size | rejected by the solver (`invalid XIR block width constraint`) |
| exact `threads_per_group` conflicting with `PlannerOptions::block_size` | rejected (`Conflicting XIR and Runtime block width constraints`) |
| static snapshot demand over the per-lane budget | rejected by the planner (`max_snapshot_bytes_per_worker`) |

Diagnostics: `LUISA_TILE_XIR2AST_DISABLE` forces rejection;
`LUISA_TILE_XIR2AST_REPORT_XIR` dumps the pre-normalization XIR;
`LUISA_TILE_XIR_MAX_LOCAL_BYTES` overrides the per-lane snapshot budget.

## GPU execution target

`GPUTileExecutionTargetInfo` (`tile_xir_kernel.h`) narrows the planner contract
for GPU backends:

- `packet_width = warp_size` (DX wave size / Vulkan subgroup size);
  block-width candidates are warp-multiple powers of two up to the device max
  (DX: 1024; VK: `maxComputeWorkGroupSize[0]` clamped to 1024).
- `supports_local_distribution() == false` in v1: kernels run with
  complete-program lanes, which guarantees no `warp_lane_id`/`WARP_READ_LANE`
  emission and avoids the unvetted collective ABI through xir2ast.
  `required_packet_width != 0` after lowering is a hard error.
- `schedule()` maps the launch grid 1:1 (one task per block, one owner) and
  never inherits CPU home chunks, work stealing, or caller-thread activation
  costs.
- Ordered reductions disable fast math (`OrderedReductionAnalysis`), matching
  the SIMD/Metal gate.
- `metadata.disjoint_writes` stays `false`: the realization performs no
  view-forwarding that would require the invocation-time noalias contract.

## VK double lowering (future work)

With `LUISA_XIR_TO_SPIR-V`, VK internally re-lowers the AST back to XIR inside
`compile_spirv`. This is semantically safe (the xir2ast round trip has
dedicated tests) and costs compile time only. A future optimization could hand
the tile-XIR module directly to `SpirvCodegenEntry::compile_spirv_xir`, but
that entry still requires an AST function for the external ABI/descriptor
bindings, so the xir2ast step remains necessary for now. No HLSL-fallback
reason applies to tile kernels (no printing/native_include/async_copy/
motion_blur), so `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV` stays satisfiable.

## Tests

`src/tests/unit/tile/bridge/test_xir_runtime_gpu.cpp` is the portable GPU
subset (pointwise map/store, copy/transpose, fold policies, serial/pipeline
nests, block-width constraints, fail-closed options, overlapping writable
views). It is registered per backend in `src/tests/CMakeLists.txt`
(`test_tile_xir_runtime_gpu_dx` / `..._vk`) and runs under the xmake test
target `test_tile_xir_runtime_gpu` via `xmake run test_tile_xir_runtime_gpu dx`.
