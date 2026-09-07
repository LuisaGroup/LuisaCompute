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
├── backends/     Plugins: CUDA, DX, Metal, Vulkan, HIP, fallback, common
├── clangcxx/     Clang-based C++→GPU shader compiler (experimental)
├── core/         Foundation: types, math, logging, platform, STL wrappers
├── dsl/          Embedded C++ DSL (kernel/callable lambda tracing)
├── ext/          Third-party deps (git submodules)
├── gui/          Windowing, ImGui, framerate
├── osl/          Open Shading Language parser
├── py/           Python bindings (pybind11 + pure Python)
├── runtime/      Unified runtime: device, buffer, image, stream, RTX, raster
├── tensor/       Tensor ops & compute graph
├── tests/        Unit/integration/example tests
├── vstl/         Virtual STL: custom containers, allocators, hashes
└── xir/          Extended IR: SSA, basic blocks, passes, translators

include/luisa/    Public headers mirroring src/ layout

Root also has: examples/, tests/, tutorials/, utils/, docs/
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
- `ast2json.cpp` — AST→JSON serialization
- `constant_data.cpp`, `callable_library.cpp`, `external_function.cpp`, `function_duplicator.cpp`, `atomic_ref_node.cpp`

### `src/xir/` — Native C++ IR
SSA IR with basic blocks, instructions, optimization passes. Receives AST via `ast2xir`.
- `instructions/` — 30+ types: arithmetic, memory, control flow, resource, autodiff, atomic
- `passes/` — DCE, mem2reg, SROA, autodiff, outline, dom-tree, GEP tracing, local load/store elimination, ray-query lowering, unused callable removal, LICM, GVN, SCCP, inlining, CFG simplification
- `translators/` — `ast2xir`, `xir2json`, `json2xir`, `xir2text`, `xir2ast`
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
- `rhi/` — `device_interface.h`, `command.h`, `command_encoder.h`, `resource.h`
- `rtx/` — `accel.cpp`, `mesh.cpp`, `curve.cpp`, `motion_instance.cpp`, `procedural_primitive.cpp`
- `raster/` — `raster.cpp`, `depth_buffer.cpp`

### `src/backends/` — Backend Plugins
Dynamically loaded (`luisa-backend-<name>.dll/.so`). Each: codegen (AST/XIR→native) + compiler (NVRTC/DXC/etc.) + resources + command encoder.

| Backend | Technology |
|---|---|
| **CUDA** (`cuda/`) | NVRTC + OptiX + CUDA driver |
| **DirectX** (`dx/`) | DX12 + DXR + HLSL DXC |
| **Metal** (`metal/`) | Metal 3 + MSL |
| **Vulkan** (`vk/`) | Vulkan + SPIR-V |
| **HIP** (`hip/`) | AMD HIP |
| **Fallback** (`fallback/`) | Native C++ LLVM JIT + Embree |
| **Common** (`common/`) | `hlsl/`, `spirv/`, `spirv_llvm/`, Vulkan swapchain helpers |
| **Validation** (`validation/`) | Debug layer |

### `src/py/` — Python
- `lcapi.cpp` — pybind11 entry; `export_*.cpp` — per-component bindings
- `luisa/` — pure Python package: `buffer.py`, `accel.py`, `autodiff.py`, `gui.py`, `types.py`
- `interop.cpp/h` — PyTorch/DLPack

### `src/tensor/` — Tensor & Compute Graph
High-level tensor ops, expression DAG, graph passes.
- `fallback/` — CPU kernels (matmul, softmax)
- `pass/` — graph passes

### `src/clangcxx/`, `src/osl/`, `src/gui/`, `src/ext/`
- **clangcxx**: Clang/libTooling C++→GPU compiler (experimental)
- **osl**: OSO bytecode parser for shader interop
- **gui**: Cross-platform windowing + ImGui
- **ext**: git submodules: EASTL, glfw, glslang, imgui, pybind11, spdlog, reproc, SPIRV-Tools, spirv-headers, stb, volk, yyjson, xxhash, marl, half, HIPRT, liblmdb, magic_enum

### `src/tests/`
- `unit/{core,ast,dsl,runtime,ext,xir}/` — unit tests by layer
- `integration/{runtime,ir}/` — cross-cutting integration tests
- `common/` — shared headers (`test_device.h`, `ut/`, asset loaders)
- `cxx_shaders/` — C++ shader tests
- `python/` — Python frontend tests
- `ut/` — extra UT harness directory
- Root: integration tests (`test_path_tracing`, `test_dsl`, `test_rtx`, `test_raster`, `test_tensor`, `test_autodiff`, etc.)

## Build System

- **CMake (primary)**: root + `src/CMakeLists.txt`, targets: `luisa-compute-<name>`, alias: `luisa::compute`. Backends as `MODULE` plugins named `luisa-backend-<name>`. Options: `LUISA_COMPUTE_ENABLE_CUDA|DX|METAL|VULKAN|HIP|FALLBACK|DSL|TENSOR|GUI|...`
- **XMake (secondary)**: `xmake.lua` in root + `src/` and subdirs
- **Bootstrap**: `bootstrap.py` at repo root
- **IntelliSense**: `update_intellisense.lua`

## Compiler Pipeline

```
DSL Tracing (src/dsl/) → AST (src/ast/) → XIR (src/xir/)
                                              │
                                              ▼
                               Backend codegen / xir2ast
                                              │
                                              ▼
                                  Runtime execution (src/runtime/)
```

## Key Headers

| Header | Scope |
|---|---|
| `<luisa/luisa-compute.h>` | Core + AST + DSL + Runtime + GUI |
| `<luisa/dsl/syntax.h>` | DSL core |
| `<luisa/dsl/sugar.h>` | Sugar macros |
| `<luisa/runtime/context.h>` | Runtime entry |
| `<luisa/runtime/device.h>` | Device & resources |

## Design Patterns

1. **RHI**: `src/runtime/rhi/` abstracts GPU APIs into common interfaces
2. **Plugin Architecture**: Backends as runtime-loaded dynamic modules
3. **RAII Resources**: Move-only handles (`Buffer`, `Image`, `Stream`, `Accel`)
4. **Command-Based**: Work encoded as `Command` → `CommandList` → `Stream`
5. **DSL Tracing**: Operator overloading + lambda capture builds AST at definition time
6. **AST + XIR**: AST frontend tree plus native C++ SSA IR and optimization passes

## Naming

| Convention | Example |
|---|---|
| CMake target | `luisa-compute-core` |
| Backend binary | `luisa-backend-cuda` |
| PCH | `lc_core_pch.h` |
| Integration test | `test_path_tracing.cpp` |
| Python export | `export_runtime.cpp` |

## Maintenance

- **New backend**: `src/backends/<name>/`, implement `DeviceInterface`, register in `src/backends/CMakeLists.txt`
- **New XIR pass**: `src/xir/passes/`, register in `src/xir/CMakeLists.txt`
- **New runtime resource**: define in `rhi/resource.h`, implement per-backend, expose in `runtime/` + `include/luisa/runtime/`
- `tensor/` is opt-in, less mature. `clangcxx/` is experimental.
