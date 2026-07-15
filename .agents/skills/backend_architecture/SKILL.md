---
name: backend_architecture
description: Backend plugin architecture, DeviceInterface, dynamic loading, and command encoding.
---

# Backend Plugin Architecture

Backends are dynamically loaded shared libraries (`luisa-backend-<name>.dll/.so/.dylib`) implementing `DeviceInterface`. Discovered and loaded at runtime by `Context`.

**Header**: `include/luisa/runtime/rhi/device_interface.h`

## DeviceInterface API

All backends inherit from `DeviceInterface`:

### Resource Lifecycle (Handle-based)
| Resource | Create | Destroy | Notes |
|---|---|---|---|
| Buffer | `create_buffer(const Type*, size_t elem_count, void* external_memory)` | `destroy_buffer(uint64_t)` | Overload also takes `const ir::CArc<ir::Type>*`; returns `BufferCreationInfo` |
| Texture | `create_texture(PixelFormat, uint dimension, w, h, d, mips, external_native_handle, simultaneous_access, allow_raster_target)` | `destroy_texture(uint64_t)` | Returns `ResourceCreationInfo` |
| Bindless Array | `create_bindless_array(size_t, BindlessSlotType)` | `destroy_bindless_array(uint64_t)` | `BindlessSlotType` selects buffer/2D/3D-only or mixed slots |
| Stream | `create_stream(StreamTag)` | `destroy_stream(uint64_t)` | Graphics/compute/copy queues |
| Event | `create_event()` | `destroy_event(uint64_t)` | Timeline events |
| Shader | `create_shader(ShaderOption, Function / ir::KernelModule* / ir_v2::KernelModule&)` | `destroy_shader(uint64_t)` | Also `load_shader(name, arg_types)` and `shader_argument_usage(handle, index)` |
| Mesh | `create_mesh(AccelOption)` | `destroy_mesh(uint64_t)` | |
| Curve | `create_curve(AccelOption)` | `destroy_curve(uint64_t)` | Optional; default impl returns invalid |
| Procedural Primitive | `create_procedural_primitive(AccelOption)` | `destroy_procedural_primitive(uint64_t)` | |
| Motion Instance | `create_motion_instance(AccelMotionOption)` | `destroy_motion_instance(uint64_t)` | Optional; default impl returns invalid |
| Accel | `create_accel(AccelOption)` | `destroy_accel(uint64_t)` | Top-level acceleration structure |
| Swapchain | `create_swapchain(SwapchainOption, stream_handle)` | `destroy_swapchain(uint64_t)` | Also `present_display_in_stream(stream, swapchain, image)` |
| Sparse Buffer | `create_sparse_buffer(...)`, `allocate_sparse_buffer_heap(...)`, `update_sparse_resources(...)` | `destroy_sparse_buffer(...)` | Optional; default implementations return invalid |
| Sparse Texture | `create_sparse_texture(...)`, `allocate_sparse_texture_heap(...)` | `destroy_sparse_texture(...)` | Optional; default implementations return invalid |

### Execution
| Method | Purpose |
|---|---|
| `dispatch(stream_handle, CommandList &&list)` | Submit commands |
| `synchronize_stream(stream_handle)` | Host-wait for stream |
| `set_stream_log_callback(stream_handle, cb)` | Per-stream logging |
| `present_display_in_stream(stream_handle, swapchain_handle, image_handle)` | Present swapchain image |

### Queries & Extensions
| Method | Purpose |
|---|---|
| `native_handle()` | Underlying API handle (CUcontext, VkDevice, etc.) |
| `compute_warp_size()` | Warp size (32 CUDA, 1 CPU/fallback) |
| `memory_granularity()` | Allocation alignment |
| `query(property)` | Device property queries |
| `extension(name)` | Device extension interface |
| `set_name(Resource::Tag, handle, name)` | Debug naming |
| `get_name(handle)` | Retrieve debug name |

### Event Sync
| Method | Purpose |
|---|---|
| `signal_event(handle, stream_handle, fence_value)` | Signal timeline event |
| `wait_event(handle, stream_handle, fence_value)` | Wait on timeline event |
| `is_event_completed(handle, fence_value)` | Poll completion |
| `synchronize_event(handle, fence_value)` | Host-wait on event |

## Dynamic Loading

**File**: `src/runtime/context.cpp`

`Context` scans the runtime directory for libraries matching `luisa-backend-*.{so|dll|dylib}` (and `libluisa-backend-*` for MinGW). Names are lower-cased, sorted, and de-duplicated.

```cpp
const BackendModule &load_backend(const luisa::string &backend_name) {
    // 1. Check installed_backends list
    // 2. Load dynamic library: luisa-backend-<name>.<ext>
    // 3. Validate: backend_version() == LUISA_COMPUTE_VERSION
    // 4. Extract: creator("create"), deleter("destroy"), backend_device_names
}
```

Device creation:
```cpp
Device Context::create_device(backend_name, settings, enable_validation) {
    auto &m = _impl->load_backend(backend_name);
    auto interface = m.creator(Context{_impl}, settings);
    interface->_backend_name = backend_name;
    auto handle = Device::Handle{interface, [deleter = m.deleter](auto p) { deleter(p); }};
    if (enable_validation) {
        auto &validation_layer = _impl->load_validation_layer();
        handle = Device::Handle{
            validation_layer.creator(Context{_impl}, std::move(handle)),
            /* validation deleter */};
    }
    return Device{std::move(handle)};
}
```

Validation can also be enabled via the `LUISA_ENABLE_VALIDATION=1` environment variable.

## Backend Registration

Every backend must export **three C functions** with `LUISA_EXPORT_API`:

```cpp
// src/backends/<name>/<name>_device.cpp

LUISA_EXPORT_API DeviceInterface *create(Context &&ctx, const DeviceConfig *config) noexcept {
    return new_with_allocator<MyBackendDevice>(std::move(ctx), config);
}
LUISA_EXPORT_API void destroy(DeviceInterface *device) noexcept {
    delete_with_allocator(device);
}
LUISA_EXPORT_API void backend_device_names(vector<string> &names) noexcept {
    names.clear();
    names.emplace_back("<name>"); // e.g., "cuda", "cpu", "dx", "vk", "metal", "hip", ...
}
```

Plus version export (`src/backends/common/export_version.inl.h`):
```cpp
LUISA_EXPORT_API int backend_version() { return LUISA_COMPUTE_VERSION; }
```

## Command Encoder (Visitor Pattern)

Commands dispatched via `MutableCommandVisitor`:

```cpp
class MyCommandEncoder : public MutableCommandVisitor {
    MyStream *_stream;
    void visit(BufferUploadCommand *cmd) noexcept override;
    void visit(BufferDownloadCommand *cmd) noexcept override;
    void visit(BufferCopyCommand *cmd) noexcept override;
    void visit(BufferToTextureCopyCommand *cmd) noexcept override;
    void visit(TextureUploadCommand *cmd) noexcept override;
    void visit(TextureDownloadCommand *cmd) noexcept override;
    void visit(TextureCopyCommand *cmd) noexcept override;
    void visit(TextureToBufferCopyCommand *cmd) noexcept override;
    void visit(ShaderDispatchCommand *cmd) noexcept override;
    void visit(AccelBuildCommand *cmd) noexcept override;
    void visit(MeshBuildCommand *cmd) noexcept override;
    void visit(CurveBuildCommand *cmd) noexcept override;
    void visit(ProceduralPrimitiveBuildCommand *cmd) noexcept override;
    void visit(MotionInstanceBuildCommand *cmd) noexcept override;
    void visit(BindlessArrayUpdateCommand *cmd) noexcept override;
    void visit(CustomCommand *cmd) noexcept override;
    void commit(CommandList::CallbackContainer &&user_callbacks) noexcept;
};
```

Architecture: each backend has a `*Stream` class owning the native stream/queue. `stream->dispatch(CommandList)` visits all commands via an encoder. User callbacks are executed after GPU work completes.

## Resource Handle Pattern

Resources are created as backend-specific classes and returned as opaque `uint64_t` handles. Buffer creation returns extra stride/size info:

```cpp
auto buffer = new_with_allocator<CUDABuffer>(size);
BufferCreationInfo info;
info.handle = reinterpret_cast<uint64_t>(buffer);
info.native_handle = reinterpret_cast<void *>(buffer->device_address());
info.element_stride = buffer->element_stride();
info.total_size_bytes = buffer->total_size_bytes();
return info;
```

For most other resources, return a `ResourceCreationInfo` (or derived type such as `ShaderCreationInfo`/`SwapchainCreationInfo`):

```cpp
ResourceCreationInfo create_texture(...) noexcept {
    auto tex = new_with_allocator<CUDATexture>(...);
    return {.handle = reinterpret_cast<uint64_t>(tex),
            .native_handle = reinterpret_cast<void *>(tex->native_handle())};
}
```

Destruction recovers typed pointer:
```cpp
void destroy_buffer(uint64_t handle) noexcept {
    auto buffer = reinterpret_cast<CUDABuffer *>(handle);
    delete_with_allocator(buffer);
}
```

## Files Per Backend

### Minimal
| File | Purpose |
|---|---|
| `<backend>_device.h/cpp` | Device class + exported C functions |
| `<backend>_stream.h/cpp` | Stream/command queue |
| `<backend>_buffer.h/cpp` | Buffer wrapper |
| `<backend>_texture.h/cpp` | Texture wrapper |
| `<backend>_shader.h/cpp` | Shader compilation & management |
| `<backend>_event.h/cpp` | Sync primitives |
| `CMakeLists.txt` / `xmake.lua` | Build config |

### Common Helpers
`<backend>_command_encoder.h/cpp` (command visitor), `<backend>_accel.h/cpp` (ray tracing), `<backend>_mesh.h/cpp`, `<backend>_curve.h/cpp`, `<backend>_proc_prim.h/cpp`, `<backend>_motion_instance.h/cpp`, `<backend>_bindless_array.h/cpp`, `<backend>_swapchain.h/cpp`, `<backend>_sparse.h/cpp` (optional).

### Shared (`src/backends/common/`)
- `default_binary_io.h/cpp` — Shader caching I/O
- `export_version.inl.h` — Version export
- `hlsl/builtin/` — HLSL builtin headers/bytecode
- `vulkan_swapchain.h/cpp` — Shared Vulkan swapchain
- `rust_device_common.h/cpp` — Shared Rust/CPU backend helpers

## CMake Patterns

```cmake
function(luisa_compute_add_backend name)
    cmake_parse_arguments(BACKEND "" "SUPPORT_DIR;BUILTIN_DIR" "SOURCES" ${ARGN})
    add_library(luisa-compute-backend-${name} MODULE ${BACKEND_SOURCES})
    target_link_libraries(luisa-compute-backend-${name} PRIVATE
        luisa-compute-ast
        luisa-compute-runtime
        luisa-compute-gui)
    if (LUISA_COMPUTE_ENABLE_DSL)
        target_link_libraries(luisa-compute-backend-${name} PRIVATE luisa-compute-dsl)
    endif ()
    add_dependencies(luisa-compute-backends luisa-compute-backend-${name})
    set_target_properties(luisa-compute-backend-${name} PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        DEBUG_POSTFIX ""
        OUTPUT_NAME luisa-backend-${name})
    install(TARGETS luisa-compute-backend-${name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if (BACKEND_SUPPORT_DIR)
        add_custom_target(luisa-compute-backend-${name}-copy-support ALL ...)
    endif ()
endfunction()
```

**Minimal (CPU)**:
```cmake
luisa_compute_add_backend(cpu SOURCES
    ../common/rust_device_common.cpp ../common/rust_device_common.h
    cpu_device.h cpu_device.cpp)
target_link_libraries(luisa-compute-backend-cpu PRIVATE
    luisa-compute-vulkan-swapchain
    luisa-compute-rust-meta
    luisa_compute_backend_impl)
```

**Fallback** (LLVM + Embree):
```cmake
find_package(LLVM CONFIG)
find_package(embree CONFIG)
if (LLVM_FOUND AND embree_FOUND)
    luisa_compute_add_backend(fallback SOURCES fallback_device.cpp ...)
    target_link_libraries(luisa-compute-backend-fallback PRIVATE
        luisa-compute-xir
        luisa-compute-vulkan-swapchain
        embree)
    luisa_compute_link_llvm_into_backend(fallback REQUIRED
        COMPONENTS core executionengine support orcjit nativecodegen irreader passes analysis coroutines)
endif()
```

**CUDA** advanced: `luisa_embed_device_lib`, device runtime embedding, optional nvCOMP/NVTT/OIDN, standalone NVRTC compiler target.

### XMake Note
The project also supports XMake builds. Backend `xmake.lua` files create shared targets named `lc-backend-<name>` with `set_basename("luisa-backend-<name>")`, link `lc-runtime`/`lc-ir`, and optionally depend on `lc-vulkan-swapchain` and backend-specific helpers.

## Codegen Pipeline

```
AST Function / XIR KernelModule / IRv2 KernelModule
                │
                ▼
┌─────────────────────────────────┐
│         Backend Codegen          │  → native shader source (CUDA C++, HLSL, MSL, SPIR-V)
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│        Native Compiler           │  NVRTC, DXC, Metal, SPIR-V tools
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│        Shader Object             │  PTX, DXIL, metallib, SPIR-V binary
│        (BinaryIO cache)          │  Cached via BinaryIO
└─────────────────────────────────┘
```

## Key Design Decisions

1. **Plugin Architecture** — Backends are shared libs loaded at runtime, distributable separately
2. **Handle-based Resources** — Opaque `uint64_t` handles for ABI stability
3. **Visitor Pattern** — Double dispatch via `MutableCommandVisitor` for type-safe command handling
4. **Version Checking** — Strict match between runtime and backend prevents ABI mismatches
5. **Validation Layer** — Optional wrap-around validation at device creation (or via `LUISA_ENABLE_VALIDATION`)
6. **BinaryIO** — Pluggable shader caching via `default_binary_io`
