---
name: project_structure
description: Project layout, module architecture, compiler pipeline, and design patterns.
---

# LuisaCompute Project Structure

Layered architecture: **Core** → **AST/XIR** → **DSL/Runtime** → **Backends**. `src/` + public headers `include/luisa/`. Dual build: CMake + XMake. Frontends: C++ and Python.

## Top-Level Directory Map

```
src/
├── ast/          AST (expressions, statements, types, function builder)
├── backends/ Plugins: CUDA, DX, Metal, Metal4, Vulkan, HIP, fallback, SIMD, remote, common, validation, tools
├── clangcxx/ Clang-based C++→GPU shader compiler (experimental)
├── core/ Foundation: types, math, logging, platform, STL wrappers
├── coro/ Coroutine (split-function) kernel schedulers
├── dsl/ Embedded C++ DSL (kernel/callable lambda tracing)
├── ext/ Third-party deps (git submodules + vendored libs)
├── gui/ Windowing, ImGui, framerate
├── osl/ Open Shading Language parser
├── py/ Python bindings (pybind11 + pure Python)
├── runtime/ Unified runtime: device, buffer, image, stream, RTX, raster
├── tests/ Unit/integration/example tests
├── tile/ Tile-level IR (TIRx/XIR bridges, layouts, verifier)
├── vstl/ Virtual STL: custom containers, allocators, hashes
└── xir/ Extended IR: SSA, basic blocks, passes, translators

include/luisa/ Public headers mirroring src/ layout (no py/, tests/, ext/; backends only as include/luisa/backends/{common,ext}/)

Root also has: examples/, tutorials/, utils/, docs/, cmake/, config/, scripts/
```

## Modules

### `src/core/` — Foundation
Platform abstractions, math, logging, binary I/O, dynamic modules.
- `basic_types.cpp` — vector/matrix instantiations
- `logging.cpp` — spdlog-based logging
- `platform.cpp` — OS abstraction (paths, threads, DLL)
- `dynamic_module.cpp` — cross-platform shared library loader
- `binary_io.cpp`, `binary_file_stream.cpp` — binary serialization
- `first_fit.cpp`, `pool.cpp`, `string_scratch.cpp` — allocators/scratch buffers
- `stl/` — custom STL: `vector`, `string`, `unordered_map`, `optional`, `variant`, etc.
- `generate_swizzles.py` — swizzle codegen

### `src/vstl/` — Virtual STL
High-perf containers beyond `core/stl`: `stack_allocator`, `string_builder`, `lmdb`, `md5`, `v_guid`. Headers: `include/luisa/vstl/*` (hash maps, arenas, lockfree queues, ranges).

### `src/ast/` — Abstract Syntax Tree
DSL traces C++ lambdas → AST nodes.
- `expression.cpp` — literal, binary, unary, call, swizzle, member
- `statement.cpp` — if, loop, switch, break, return, ray_query
- `type.cpp` — scalars, vectors, matrices, buffers, textures, structs
- `function.cpp` — kernel/callable metadata
- `function_builder.cpp` — manual AST construction API
- `variable.cpp` — local variables
- `op.cpp` — `BinaryOp`, `UnaryOp`, `CallOp`
- `ast2json.cpp`, `json2ast.cpp` — AST↔JSON serialization
- `coro_suspend.cpp` — coroutine suspend node
- `constant_data.cpp`, `callable_library.cpp`, `external_function.cpp`, `function_duplicator.cpp`, `atomic_ref_node.cpp`

### `src/xir/` — Native C++ IR
SSA IR with basic blocks, instructions, optimization passes. Receives AST via `ast2xir`.
- `instructions/` — 31 files: arithmetic, memory, control flow, resource, autodiff, atomic, coro, ray_query, raster_discard, thread_group
- `passes/` — 89 pass sources: DCE, mem2reg, SROA, autodiff, outline, dom-tree, GEP tracing, local load/store elimination, ray-query lowering, unused callable removal, LICM, GVN, SCCP, inlining, CFG simplification (plus coro_*, loop_*, vectorization, alias/uniformity analysis)
- `translators/` — `ast2xir`, `xir2json`, `json2xir`, `xir2text`, `xir2ast`, `xir_interchange`
- `metadata/` — source locations, names, comments, curve basis
- `tests/` — XIR unit tests (enabled by `LUISA_COMPUTE_ENABLE_XIR_TESTS`)
- Key classes: `Module`, `Function`, `BasicBlock`, `Instruction`, `Value`, `Use`, `Builder`

### `src/dsl/` — Embedded DSL
GPU kernels via lambda tracing.
- `func.cpp` — `Kernel1D/2D/3D`, `Callable`
- `builtin.cpp` — `dispatch_id`, `thread_id`, math
- `resource.cpp` — buffer/image/volume/bindless DSL wrappers
- `local.cpp` — local/thread storage helpers
- `sugar.cpp` — `$if`, `$for`, `$while`
- `rtx/` — ray tracing: `Accel`, `Ray`, `RayQuery`, `Curve`, `TriangleHit`
- `raster/` — `RasterKernel`
- `ext/` — DSL extensions
- `soa.cpp`, `polymorphic.cpp`, `dispatch_indirect.cpp`

### `src/runtime/` — Unified Runtime
Resource management, command scheduling, RHI abstraction.
- `device.cpp`, `context.cpp` — device creation, backend loading
- `stream.cpp`, `command_list.cpp` — command batching/submission
- `buffer.cpp`, `image.cpp`, `volume.cpp` — GPU memory
- `byte_buffer.cpp`, `dispatch_buffer.cpp`, `mipmap.cpp` — auxiliary buffers
- `sparse_buffer.cpp`, `sparse_texture.cpp`, `sparse_heap.cpp`, `sparse_command_list.cpp` — sparse resources
- `bindless_array.cpp`, `swapchain.cpp`, `event.cpp`, `builtin_kernel.cpp`
- `rhi/` — `command.cpp`, `command_encoder.cpp`, `device_interface.cpp`, `resource.cpp`, `pixel.cpp` (headers: `include/luisa/runtime/rhi/`)
- `rtx/` — `accel.cpp`, `mesh.cpp`, `curve.cpp`, `motion_instance.cpp`, `procedural_primitive.cpp`
- `raster/` — `raster.cpp`, `depth_buffer.cpp`

### `src/backends/` — Backend Plugins
Dynamically loaded (`luisa-backend-<name>.dll/.so`). Each: codegen (AST/XIR→native) + compiler (NVRTC/DXC/etc.) + resources + command encoder.

| Backend | Technology |
|---|---|
| **CUDA** (`cuda/`) | NVRTC + OptiX + CUDA driver (`cuda_codegen_ast.h`, `cuda_codegen_xir.h`) |
| **DirectX** (`dx/`) | DX12 + DXR + HLSL DXC |
| **Metal** (`metal/`) | Metal 3 + MSL (`metal_codegen_ast.h`) |
| **Metal4** (`metal4/`) | Metal 4 AIR via LLVM codegen (`llvm_codegen/`, `metal_xir_pipeline.cpp`) |
| **Vulkan** (`vk/`) | Vulkan + SPIR-V (native XIR→SPIR-V, or experimental AST→LLVM→SPIR-V) |
| **HIP** (`hip/`) | AMD HIP + HIPRT ray tracing |
| **Fallback** (`fallback/`) | Native C++ LLVM JIT (`ExecutionEngine`) + Embree |
| **SIMD** (`simd/`) | Experimental CPU backend: XIR → schedule → LLVM (Embree packets) |
| **Remote** (`remote/`) | Experimental client/server backend (`remote_protocol.cpp`, `remote_server.cpp`) |
| **Common** (`common/`) | `hlsl/`, `spirv/`, `spirv_llvm/`, `native_shader/`, `rtx/`, `llvm_native_math.*`, Vulkan swapchain/instance, OIDN denoiser |
| **Validation** (`validation/`) | Debug layer |
| **Tools** (`tools/`) | Host AOT tool `lc_compile_builtin` (`main.cpp`), not a plugin backend |

**Native shader injection** (`NativeShaderExt`): `include/luisa/backends/ext/native_shader_ext.h` (public API),
`src/backends/common/native_shader/native_shader_reflection.h` (shared SPIR-V reflection parser),
`src/backends/dx/DXApi/native_shader_ext.{h,cpp}` (HLSL→DXIL + DXC reflection),
`src/backends/vk/native_shader_ext.{h,cpp}`, `native_shader.{h,cpp}`, `glslang_compiler.{h,cpp}`
(GLSL/HLSL→SPIR-V, Tier-B pipeline built from reflection), `src/backends/validation/native_shader_ext_impl.*`,
`src/backends/cuda/native_shader_ext.{h,cpp}`, `native_shader.{h,cpp}`, `native_shader_reflection.h`
(CUDA C++ -> PTX via NVRTC + driver-API module load; reflection from the `__global__` signature crossed
with the compiled PTX parameter layout),
`examples/compute/native_shader.cpp`, and the `test_native_shader*` tests. Dispatch goes through
`CustomCommandUUID::NATIVE_SHADER_DISPATCH` (`NativeShaderDispatchCommand`), so the reorder pass and the per-backend
barriers use the declared per-argument `Usage`.

### `src/py/` — Python
- `lcapi.cpp` — pybind11 entry (`pybind11_add_module(lcapi ...)`); `export_*.cpp` — per-component bindings
- `luisa/` — pure Python package: `buffer.py`, `accel.py`, `autodiff.py`, `gui.py`, `types.py`
- `interop.cpp/h` — DX/VK↔CUDA device interop; `export_dlpack.cpp` + `dlpack.h` — DLPack (PyTorch) import/export

### `src/tile/` — Tile-level IR
Tile program IR, layouts, targets and verification (CMake target `luisa-compute-tile`, headers `include/luisa/tile/*`).
- `ir.cpp`, `layout.cpp`, `dimension.cpp`, `target.cpp`, `dsl.cpp`, `verifier.cpp`
- `bridge/xir/` — TileIR↔XIR bridge; `bridge/tirx/` — optional TVM TIRx bridge (`LUISA_COMPUTE_ENABLE_TILE_TIRX_BRIDGE`)

### `src/coro/` — Coroutine kernels
Split-function / coroutine kernel scheduling on top of DSL + XIR (CMake target `luisa-compute-coro`, headers `include/luisa/coro/*`).
- `coro_compile.cpp`, `coro_graph.cpp`, `coro_stub.cpp`
- `schedulers/` — `wavefront.cpp`, `graph_wavefront_policy.cpp`, `state_machine.cpp`, `persistent.cpp`

### `src/clangcxx/`, `src/osl/`, `src/gui/`, `src/ext/`
- **clangcxx**: Clang/libTooling C++→GPU compiler (experimental)
- **osl**: OSO bytecode parser for shader interop
- **gui**: Cross-platform windowing + ImGui
- **ext**: git submodules (`.gitmodules`): EASTL, glfw, glslang, imgui, pybind11, spdlog, reproc, SPIRV-Tools, spirv-headers, stb, yyjson, xxhash, marl, HIPRT, magic_enum, llvm-downgrade, tvm; vendored in-tree (not submodules): `volk/`, `half/`, `liblmdb/`, `BTree/`, `tvm_ext/`

### `src/tests/`
- `unit/{core,ast,dsl,runtime,xir,coro,tile,ext,fallback,simd}/` — unit tests by layer
- `integration/runtime/` — cross-cutting tests (`test_rtx.cpp`, `test_raster.cpp`, `test_aot.cpp`, `test_metal_xir_air*.cpp`, …); `integration/xir/` — `test_xir2ast_roundtrip.cpp`
- `benchmark/`, `cuda/`, `ios/` — benchmark and platform-specific tests
- `common/` — shared headers (`test_device.h`, `tiny_obj_loader.h`, `tinyexr.h`, `*_test_utils.h`)
- `cxx_shaders/` — C++ shader tests
- `python/` — Python frontend tests (`test-*.py`)
- `ut/` — vendored Boost.UT single header (`ut.hpp`)
- No test sources at the `src/tests/` root; all live in the directories above (`luisa_compute_add_test(name source)` in `src/tests/CMakeLists.txt`) 

## Build System

- **CMake (primary)**: root + `src/CMakeLists.txt`, targets: `luisa-compute-<name>` (`core`, `vstl`, `tile`, `ast`, `xir`, `runtime`, `dsl`, `coro`, `osl`, `gui`, `backends`, `clangcxx`), aggregate interface `compute` with alias: `luisa::compute`. Backends built by `luisa_compute_add_backend(<name>)` as `MODULE` plugins (`STATIC` on iOS) with `OUTPUT_NAME luisa-backend-<name>`. Options: `LUISA_COMPUTE_ENABLE_CUDA|DX|METAL|METAL4|VULKAN|HIP|FALLBACK|SIMD|REMOTE|DSL|GUI|CLANG_CXX|SAFE_MODE|...` (there is no `..._ENABLE_TENSOR`; the tensor module was removed). Outputs: `${CMAKE_BINARY_DIR}/bin` (+ `/bin/<config>` for multi-config), libs to `${CMAKE_BINARY_DIR}/lib` (`scripts/setup_output_dirs.cmake`).
- **XMake (secondary)**: `xmake.lua` in root + `src/` and subdirs; module targets `lc-*` (e.g. `lc-runtime`), plugin targets `lc-backend-{dx,cuda,metal,vk,fallback}` — hip/metal4/simd/remote have no xmake targets, CMake only. Options declared as `option("lc_*")` in the root `xmake.lua` (e.g. `lc_dx_backend`, `lc_vk_backend`, `lc_cuda_backend`, `lc_metal_backend`, `lc_fallback_backend`, `lc_enable_dsl`, `lc_enable_simd`); `lc_enable_tensor` does not exist.
- **Bootstrap**: `bootstrap.py` at repo root
- **IntelliSense**: `update_intellisense.lua`

## Compiler Pipeline

```
DSL Tracing (src/dsl/, detail::FunctionBuilder)
        │
        ▼
AST (src/ast/)
        │
        ▼
ast2xir (src/xir/translators/ast2xir.cpp)
        │
        ▼
XIR Module / CFG (src/xir/) + PassPipeline (src/xir/passes/)
        │
        ▼  Backend codegen
          ├── DX: AST → HLSL (src/backends/common/hlsl/hlsl_codegen.cpp) → DXC
          ├── Metal: AST → MSL (src/backends/metal/metal_codegen_ast.cpp)
          ├── Vulkan: XIR → SPIR-V (src/backends/common/spirv/spirv_codegen/);
                 experimental AST → LLVM → SPIR-V (src/backends/common/spirv_llvm/)
          ├── CUDA / HIP: AST + XIR → device C++ → PTX via NVRTC
                 (cuda_codegen_ast.cpp, cuda_codegen_xir.cpp)
          ├── Metal4: XIR → AIR via LLVM (src/backends/metal4/llvm_codegen/)
          ├── Fallback / SIMD: XIR → LLVM (JIT / CPU packet schedule)
          └── xir2ast (src/xir/translators/xir2ast.cpp): XIR → AST, used by the DX/VK tile paths
        │
        ▼
Runtime execution (src/runtime/)
```

## Key Headers

| Header | Scope |
|---|---|
| `<luisa/luisa-compute.h>` | Universal header: Core + vstl + AST + XIR + tile + Runtime + osl + DSL/GUI/clangcxx (the last three behind `LUISA_ENABLE_DSL` / `LUISA_ENABLE_GUI` / `LUISA_ENABLE_CLANGCXX`) |
| `<luisa/dsl/syntax.h>` | DSL core |
| `<luisa/dsl/sugar.h>` | Sugar macros |
| `<luisa/runtime/context.h>` | Runtime entry |
| `<luisa/runtime/device.h>` | Device & resources |

## Design Patterns

1. **RHI**: `include/luisa/runtime/rhi/` (`device_interface.h`, `command.h`, `command_encoder.h`, `resource.h`; impl in `src/runtime/rhi/`) abstracts GPU APIs into common interfaces
2. **Plugin Architecture**: Backends as runtime-loaded dynamic modules — `Context::load_backend()` / `DynamicModule::load(..., "luisa-backend-<name>")` in `src/runtime/context.cpp`
3. **RAII Resources**: Move-only handles (`Buffer`, `Image`, `Stream`, `Accel`)
4. **Command-Based**: Work encoded as `Command` → `CommandList` → `Stream`
5. **DSL Tracing**: Operator overloading + lambda capture builds AST at definition time
6. **AST + XIR**: AST frontend tree plus native C++ SSA IR and optimization passes

## Naming

| Convention | Example |
|---|---|
| CMake target | `luisa-compute-core` |
| Backend binary | `luisa-backend-cuda` |
| PCH | `lc_dsl_pch.h` (also `lc_runtime_pch.h`, `lc_vk_pch.h`, …; there is no `lc_core_pch.h`) |
| Integration test | `test_rtx.cpp` (`src/tests/integration/runtime/`) |
| Python export | `export_runtime.cpp` |

## Maintenance

- **New backend**: `src/backends/<name>/`, implement `DeviceInterface`, add `add_subdirectory(<name>)` behind a `LUISA_COMPUTE_ENABLE_<NAME>` guard in `src/backends/CMakeLists.txt` and call `luisa_compute_add_backend(<name> ...)` there
- **New XIR pass**: `src/xir/passes/`, register in `src/xir/CMakeLists.txt`
- **New runtime resource**: define in `include/luisa/runtime/rhi/resource.h`, implement per-backend, expose in `runtime/` + `include/luisa/runtime/`
- `hip/`, `metal4/`, `simd/`, `remote/` backends and `clangcxx/` are OFF by default (CMake) and experimental. There is no `tensor/` module any more (`add_subdirectory(tensor)` is commented out in `src/CMakeLists.txt` and `src/xmake.lua`); `examples/tensor/` is a hand-written DSL example.
