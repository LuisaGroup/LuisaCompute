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
| Resource | Create | Destroy |
|---|---|---|
| Buffer | `create_buffer(Type*, size, external)` | `destroy_buffer(uint64_t)` |
| Texture | `create_texture(format, dim, w, h, d, mips, ...)` | `destroy_texture(uint64_t)` |
| Bindless Array | `create_bindless_array(size, type)` | `destroy_bindless_array(uint64_t)` |
| Stream | `create_stream(StreamTag)` | `destroy_stream(uint64_t)` |
| Event | `create_event()` | `destroy_event(uint64_t)` |
| Shader | `create_shader(ShaderOption, Function/IR)` | `destroy_shader(uint64_t)` |
| Mesh | `create_mesh(AccelOption)` | `destroy_mesh(uint64_t)` |
| Curve | `create_curve(AccelOption)` | `destroy_curve(uint64_t)` |
| Procedural Primitive | `create_procedural_primitive(AccelOption)` | `destroy_procedural_primitive(uint64_t)` |
| Motion Instance | `create_motion_instance(AccelMotionOption)` | `destroy_motion_instance(uint64_t)` |
| Accel | `create_accel(AccelOption)` | `destroy_accel(uint64_t)` |
| Swapchain | `create_swapchain(SwapchainOption, stream)` | `destroy_swapchain(uint64_t)` |

### Execution
| Method | Purpose |
|---|---|
| `dispatch(stream_handle, CommandList)` | Submit commands |
| `synchronize_stream(stream_handle)` | Host-wait for stream |
| `set_stream_log_callback(stream_handle, cb)` | Per-stream logging |

### Queries & Extensions
| Method | Purpose |
|---|---|
| `native_handle()` | Underlying API handle (CUcontext, VkDevice, etc.) |
| `compute_warp_size()` | Warp size (32 CUDA, 1 CPU/fallback) |
| `memory_granularity()` | Allocation alignment |
| `query(property)` | Device property queries |
| `extension(name)` | Device extension interface |
| `set_name(handle, name)` | Debug naming |

### Event Sync
| Method | Purpose |
|---|---|
| `signal_event(handle, stream, value)` | Signal timeline event |
| `wait_event(handle, stream, value)` | Wait on timeline event |
| `is_event_completed(handle, value)` | Poll completion |
| `synchronize_event(handle, value)` | Host-wait on event |

## Dynamic Loading

**File**: `src/runtime/context.cpp`

`Context` scans runtime dir for `luisa-backend-*.so|.dll|.dylib` (or `libluisa-backend-*` on MinGW).

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
    if (enable_validation) interface = wrap_with_validation(interface);
    return Device{handle};
}
```

## Backend Registration

Every backend must export **three C functions** with `LUISA_EXPORT_API`:

```cpp
// src/backends/<name>/<name>_device.cpp

LUISA_EXPORT_API DeviceInterface *create(Context &&ctx, const DeviceConfig *) noexcept {
    return new_with_allocator<MyBackendDevice>(std::move(ctx));
}
LUISA_EXPORT_API void destroy(DeviceInterface *device) noexcept {
    delete_with_allocator(device);
}
LUISA_EXPORT_API void backend_device_names(vector<string> &names) noexcept {
    names.clear();
    names.emplace_back("cuda"); // or "cpu", "dx", "vk", ...
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
    void visit(ShaderDispatchCommand *cmd) noexcept override;
    void visit(AccelBuildCommand *cmd) noexcept override;
    void visit(MeshBuildCommand *cmd) noexcept override;
    void visit(BindlessArrayUpdateCommand *cmd) noexcept override;
    void visit(CustomCommand *cmd) noexcept override;
    void commit(CommandList::CallbackContainer &&user_callbacks) noexcept;
};
```

Architecture: each backend has a `*Stream` class owning the native stream/queue. `stream->dispatch(CommandList)` visits all commands via encoder. Callbacks executed after GPU work completes.

## Resource Handle Pattern

Resources created as backend-specific classes, returned as opaque `uint64_t`:

```cpp
auto buffer = new_with_allocator<CUDABuffer>(size);
return {.handle = reinterpret_cast<uint64_t>(buffer),
        .native_handle = reinterpret_cast<void *>(buffer->device_address())};
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
| `CMakeLists.txt` | Build config |

### Common Helpers
`<backend>_command_encoder.h/cpp` (command visitor), `<backend>_accel.h/cpp` (ray tracing), `<backend>_mesh.h/cpp`, `<backend>_bindless_array.h/cpp`, `<backend>_swapchain.h/cpp`

### Shared (`src/backends/common/`)
- `default_binary_io.h/cpp` — Shader caching I/O
- `export_version.inl.h` — Version export
- `hlsl/builtin/` — HLSL builtin headers/bytecode
- `vulkan_swapchain.h/cpp` — Shared Vulkan swapchain

## CMake Patterns

```cmake
function(luisa_compute_add_backend name)
    cmake_parse_arguments(BACKEND "" "SUPPORT_DIR;BUILTIN_DIR" "SOURCES" ${ARGN})
    add_library(luisa-compute-backend-${name} MODULE ${BACKEND_SOURCES})
    target_link_libraries(luisa-compute-backend-${name} PRIVATE
        luisa-compute-ast luisa-compute-runtime luisa-compute-gui)
    set_target_properties(luisa-compute-backend-${name} PROPERTIES
        OUTPUT_NAME luisa-backend-${name})
    install(TARGETS luisa-compute-backend-${name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()
```

**Minimal (CPU)**:
```cmake
luisa_compute_add_backend(cpu SOURCES
    ../common/rust_device_common.cpp ../common/rust_device_common.h
    cpu_device.h cpu_device.cpp)
target_link_libraries(luisa-compute-backend-cpu PRIVATE
    luisa-compute-vulkan-swapchain luisa-compute-rust-meta luisa_compute_backend_impl)
```

**Fallback** (LLVM + Embree):
```cmake
find_package(LLVM CONFIG)
find_package(embree CONFIG)
if (LLVM_FOUND AND embree_FOUND)
    luisa_compute_add_backend(fallback SOURCES fallback_device.cpp ...)
    target_link_libraries(luisa-compute-backend-fallback PRIVATE
        luisa-compute-xir luisa-compute-vulkan-swapchain embree)
    luisa_compute_link_llvm_into_backend(fallback REQUIRED
        COMPONENTS core executionengine ...)
endif()
```

**CUDA** advanced: `luisa_embed_device_lib`, device runtime embedding, optional nvCOMP/NVTT/OIDN, standalone compiler.

## Codegen Pipeline

```
AST/XIR Function
       │
       ▼
┌─────────────────┐
│  Backend Codegen │  → native shader source (CUDA C++, HLSL, MSL, SPIR-V)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Native Compiler  │  NVRTC, DXC, Metal, SPIR-V tools
└────────┬────────┘
         ▼
┌─────────────────┐
│  Shader Object   │  PTX, DXIL, metallib, SPIR-V binary
│  (BinaryIO)      │  Cached via BinaryIO
└─────────────────┘
```

## Key Design Decisions

1. **Plugin Architecture** — Backends are shared libs loaded at runtime, distributable separately
2. **Handle-based Resources** — Opaque `uint64_t` handles for ABI stability
3. **Visitor Pattern** — Double dispatch via `MutableCommandVisitor` for type-safe command handling
4. **Version Checking** — Strict match between runtime and backend prevents ABI mismatches
5. **Validation Layer** — Optional wrap-around validation at device creation
6. **BinaryIO** — Pluggable shader caching via `default_binary_io`
