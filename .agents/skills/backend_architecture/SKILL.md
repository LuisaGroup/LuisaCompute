---
name: backend-architecture
description: Backend plugin architecture, DeviceInterface, dynamic loading, and command encoding.
---

# Backend Plugin Architecture

On desktop platforms, backends are dynamically loaded shared libraries
(`luisa-backend-<name>.dll/.so/.dylib`) implementing `DeviceInterface` and
discovered by `Context`. iOS links the selected backend and core runtime as
static slices into the signed app; it calls an explicit static create/destroy
bridge while preserving the same `DeviceInterface` ownership contract.

**Header**: `include/luisa/runtime/rhi/device_interface.h`

## DeviceInterface API

All backends inherit from `DeviceInterface`:

### Resource Lifecycle (Handle-based)
| Resource | Create | Destroy | Notes |
|---|---|---|---|
| Buffer | `create_buffer(const Type*, size_t elem_count, void* external_memory)` | `destroy_buffer(uint64_t)` | Returns `BufferCreationInfo` |
| Texture | `create_texture(PixelFormat, uint dimension, w, h, d, mips, external_native_handle, simultaneous_access, allow_raster_target)` | `destroy_texture(uint64_t)` | Returns `ResourceCreationInfo` |
| Bindless Array | `create_bindless_array(size_t, BindlessSlotType)` | `destroy_bindless_array(uint64_t)` | `BindlessSlotType` selects buffer/2D/3D-only or mixed slots |
| Stream | `create_stream(StreamTag)` | `destroy_stream(uint64_t)` | Graphics/compute/copy queues |
| Event | `create_event()` | `destroy_event(uint64_t)` | Timeline events |
| Shader | `create_shader(ShaderOption, Function)` | `destroy_shader(uint64_t)` | Also `load_shader(name, arg_types)` and `shader_argument_usage(handle, index)` |
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
| `memory_granularity()` | Legacy backend-wide allocation granularity; not a resource-specific sparse page size |
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
    names.emplace_back("<name>"); // e.g., "cuda", "dx", "vk", "metal", "hip", "fallback", ...
}
```

Plus version export (`src/backends/common/export_version.inl.h`):
```cpp
LUISA_EXPORT_API int backend_version() { return LUISA_COMPUTE_VERSION; }
```

### Static iOS registration

iOS cannot discover an in-bundle backend through desktop `MODULE` loading.
The app calls `luisa_compute_metal4_register_static_backend()` from
`src/backends/metal4/metal_static_backend.h` before device creation. That
bridge registers the normal create/destroy/device-name functions through
`Context::register_static_backend("metal4", ...)`; `Context` checks this
case-insensitive registry before dynamic loading. Keep
`create_device("metal4")`, ordinary `DeviceInterface` ownership, and the
validation-layer path intact instead of constructing a backend directly in a
UIKit host.

Each iOS bundle is a static application closure (`BUILD_SHARED_LIBS=OFF`):
runtime, DSL, XIR, Metal4, llvm-downgrade, and target LLVM archives are inside
the arm64 Mach-O. Audit with `scripts/audit_ios_bundles.sh`; `otool -L` should
show Apple system paths only.

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

### Metal4 acceleration capability boundary

Creating an `MTL4::Compiler`, queue, or AIR pipeline does not prove that every
MTL4 encoder feature is executable. Address-driven acceleration-structure
builds and component motion require Apple9. Query the concrete device family:
Apple9+ uses MTL4 primitive/instance descriptors and its compute encoder;
Apple7/Apple8 synchronize only AS build/refit/compact through an isolated
legacy `MTL::CommandQueue`. User shaders, PSOs, argument tables, command
buffers, and dispatch remain MTL4 AIR on both paths.

`MotionInstanceBuildCommand` is host-state capture: validate a built child and
copy its matrix/SRT keyframes into the backend resource. Native motion TLAS
packing occurs later in `AccelBuildCommand`. Preserve the shader-visible
72-byte static instance ABI while creating separate 48-byte indirect-motion
records and a transform buffer for the build descriptor. Matrix motion is
available where primitive motion blur is reported; component/SRT motion must
be rejected before resource creation below Apple9.

On iOS, do not call
`MTL::Device::isDepth24Stencil8PixelFormatSupported()` merely because
metal-cpp's safe-send helper finds a method signature. Some AGX devices expose
that signature without responding to the selector. Check real Objective-C
class selector responsiveness first, then map logical D24S8 storage to
D32S8A24 when unavailable; preserve the public logical format and execute both
stencil paths.

### Command reordering and bindless hazards

`src/backends/common/command_reorder_visitor.h` plans command layers from the
resource accesses visible at each command boundary. A backend instantiating
this visitor must preserve these contracts:

- Saved shader argument `Usage` is authoritative. Do not rediscover or
  approximate read/write access in the reorder or barrier layer.
- A bindless dispatch reads the bindless index/descriptor object itself.
- Snapshot every resource currently reachable from the bindless array when
  the dispatch is visited. Register each buffer as a whole-resource read or
  write according to the saved bindless argument usage. Bindless textures are
  sampled-only in the current runtime ABI and remain reads.
- The snapshot belongs to that dispatch. Never infer an earlier dispatch's
  hazards from the array's later membership; a subsequent update may replace
  a slot, and two distinct arrays may still reference the same resource.
- Backend resource-barrier preprocessing must mirror the same exact saved
  usage and resource snapshot. Correct reordering without matching native API
  access masks is insufficient, and vice versa.
- A Vulkan `VKCustomCmd::ResourceUsage` that names a bindless array applies
  its declared native stage/access/layout contract both to the array's index
  storage and to every member in the encoded descriptor snapshot at that
  command boundary. Reordering and native barriers must describe the same
  aggregate access.
- `VKCustomCmd` states are exact native contracts, not abstract usage hints.
  Place them in an isolated resource reorder layer: otherwise a read-only
  custom layout can merge with another read into one layer even though the
  two commands pass different `VkImageLayout` values to Vulkan. Isolation
  covers the bindless descriptor object and every snapshotted member.
- The simultaneous-access optimization may select `GENERAL` only for
  backend-owned commands that query the tracker's selected layout. Never
  replace a custom command's explicit native layout. When aggregating access,
  an empty prior access is an identity and repeated equal read layouts remain
  equal; only incompatible abstract read layouts collapse to `GENERAL`.
- A `VulkanDeviceConfigExt` `before_states` or `after_states` entry naming a
  bindless array expands over the descriptor index storage and the exact
  encoded member snapshot. Preserve the entry's raw stage/access contract for
  every member and its texture layout for every referenced mip; the texture
  layout must be defined whenever the snapshot contains an image.
- `VulkanDeviceConfigExt::before_states()` is expanded against the bindless
  snapshot at command-list entry. `after_states()` is expanded only after all
  commands have been encoded, against the final snapshot after in-list
  updates. Treating both callbacks as if they saw the same membership loses
  either newly inserted resources or resources removed during the list.

When a backend permits multiple opaque handles to wrap the same native
resource, the native resource identity is an additional alias boundary. Either
canonicalize hazards by that identity or enforce a documented external-sync
contract; pointer equality between backend wrappers is not enough.

For Vulkan-owned buffers and images, queue-family sharing is a separate
creation-time contract. If graphics, compute, and copy select more than one
unique queue family, create resources in concurrent sharing mode over exactly
those unique families; a single-family device may retain exclusive sharing.
Timeline events still establish execution and memory ordering between streams:
concurrent sharing removes queue-family ownership transfers, not synchronization.
Sparse resources follow the same graphics/compute/copy sharing plan. A
backend-owned logical device additionally requests one canonical queue whose
family advertises `VK_QUEUE_SPARSE_BINDING_BIT`, preferring a family without
graphics or compute capability and falling back to a shared stream family.
Route every `vkQueueBindSparse` through that queue and its canonical per-handle
mutex. Sparse binding changes page tables but does not access the resource in a
pipeline stage, so a dedicated sparse-only family is not added to the
resource-access sharing list. Imported logical devices keep sparse support
disabled until the import ABI can attest enabled sparse features and supply an
actual sparse queue handle/family; physical-device support alone is not enough.
Creation also requires the corresponding enabled logical-device features
(`sparseBinding` plus buffer/2D/3D residency), and the format-specific sparse
image query must succeed. Vulkan forbids sparse-residency 1D images.
Do not encode a guessed sparse page size or image tile shape. Query each created
resource's `VkMemoryRequirements`, query every sparse image's
`VkSparseImageMemoryRequirements`, and retain the selected color-aspect
requirement with the resource. The non-tail image tile shape is
`formatProperties.imageGranularity`; each tile consumes one full
`VkMemoryRequirements::alignment` block even when an edge bind clips its texel
extent against the selected mip. Reject mip-tail levels in the public tile-bind
path, and reject creation when any requested mip begins at or beyond
`imageMipTailFirstLod`: those levels require opaque binds that
`SparseTextureMapOperation` cannot express, so returning a partly usable image
would make the public mip range dishonest.
Likewise reject metadata or ambiguous aspect requirements until the public API
can describe their opaque bindings.

Sparse-buffer binds use `VkMemoryRequirements::alignment` for resource offset,
memory offset, and bind size. Preserve the API-visible `VkBufferCreateInfo`
size and use `VkMemoryRequirements::size` as the aligned sparse virtual-address
range; the requirements already include any padding needed by the final page.
Validate map and unmap ranges with checked arithmetic against this sparse tile
grid; do not clip a final buffer page as if it were an image edge.

The public sparse heap is allocated before a consuming Vulkan resource exposes
its memory type mask and alignment. Defer its physical `VkDeviceMemory`
allocation until the first map, allocate for that resource's real requirements,
exclude lazily allocated memory types, and require all later consumers to be
compatible with the chosen memory type and alignment. Until that first map no
`VkDeviceMemory` exists, and the immutable
`ResourceCreationInfo::native_handle` remains null: native sparse-heap interop
is intentionally unsupported on Vulkan. A map region consumes `tile_count *
alignment` bytes even when its final image tile is texel-clipped; heap-capacity
checks must use that full-block size rather than copied pixel size.
Within one `VkBindSparseInfo`, reject overlapping ranges of the same resource,
including a map overlapping an unmap. The public sparse API supplies no heap
offset or alias opt-in, and the Vulkan device does not enable aliased sparse
residency. Maintain a mutex-protected, Device-wide residency registry across
all streams: a heap may back only one live mapping (possibly split into
fragments by partial unmaps), mapped resource ranges may not overlap, and a
heap released by an unmap may not be recycled by another bind in the same
unordered batch. Require maps to target an inactive registered heap and an
entirely unmapped resource range. Require unmaps to be fully covered by live
mappings, subtract partial buffer intervals or image texel boxes exactly, and
keep the heap active until its last fragment is removed. Hold the registry
transaction through native submission and host completion, commit only after
success, and reject destruction of a heap or sparse resource with live
mappings.

Sparse-binding submissions also have no
implicit execution dependency with adjacent command-buffer submissions on the
same queue. A stream can also contain external-event wait/signal submissions
that do not advance its internal timeline. Immediately before sparse binding,
submit a bridge signal of the internal timeline on the ordinary stream queue;
make the sparse queue wait for that exact bridge value and signal a second
value. Query the current semaphore counter and reject a handoff whose farther
signal would exceed the physical device's
`maxTimelineSemaphoreValueDifference`; host-only logical fence gaps are not a
license to violate the timeline value limit. Establish an explicit completion boundary before later stream work may
be submitted. The current Vulkan backend host-waits that second value because
sparse updates are rare and the public API exposes no asynchronous residency
lifetime token; do not replace either edge with inferred submission order.

The same timeline-value window applies to ordinary `vkQueueSubmit` event waits
and signals, not only `vkQueueBindSparse`. Validate public and internal event
values through the common timeline planner. Keep a GPU-only completion
watermark so the hot dispatch path can validate conservatively without calling
`vkGetSemaphoreCounterValue` every time; query the driver only on a
limit-boundary slow path. Logical stream fences may include callback-only or
skipped-present work, so an embedded Vulkan wait (for example the standalone
present handoff) must use the last value actually submitted to the semaphore,
never the last logical fence.
The image `simultaneous_access` flag controls the layout/access policy and is
not a substitute for queue-family sharing. Imported native resources retain
their external creation-time sharing contract, while aliases of a Luisa-owned
native image must share its canonical per-mip layout state.

An imported Vulkan logical device must provide the exact family index for each
graphics, compute, and copy queue. A `VkQueue` handle cannot be queried for its
family after creation, so rediscovering a preferred family from the physical
device is not equivalent. The graphics family must support both graphics and
compute because a Luisa graphics stream records both command classes. It must
also provide all three actual queue handles: calling `vkGetDeviceQueue` would
otherwise assume queue index zero was present in the imported device's queue
creation infos. Roles may share one handle; equal handles share one backend
mutex because Vulkan host synchronization is per `VkQueue` identity.
Every path that may submit through a `VulkanDeviceConfigExt` callback must hold
that same canonical queue mutex while invoking the callback; custom submission
is not outside Vulkan's host-synchronization rules. A callback that handles an
event signal must still advance the backend's timeline-fence bookkeeping.
Returning true from a queue semaphore callback means the requested operation
was submitted on the supplied queue before return and has independent forward
progress; returning true from the host-sync callback means the requested value
is already complete. A callback must not report mere intent or defer submission
to work that depends on the blocked backend call.
`borrow_command_buffer()` is a one-shot ownership boundary: every non-null
result must be a fresh primary command buffer in the initial state for the
selected stream family, and its owner must keep both it and its pool alive
until stream completion. The backend records it but never resets, frees, or
recycles it; only backend-owned command buffers enter the reset/reuse pool.

An imported logical device must attest that `timelineSemaphore` and
`synchronization2` were enabled in its creation feature chain. Physical-device
queries prove only support, not logical-device enablement. All optional
logical-device features remain disabled on imports until the public import
contract carries a corresponding enabled-feature attestation; never infer
enablement from physical support. A borrowed instance must also report the
effective API version used at `vkCreateInstance` and must be Vulkan 1.3 or
newer. It is stored on that backend device and remains caller-owned; it must
never be installed in the process-owned internal-instance slot.
All imported Vulkan handles are borrowed, including the instance, physical
device, logical device, and queues. Their complete ancestry and originating
loader must outlive the Luisa Vulkan `Device`; backend teardown must finish
before the importer invalidates any of them.
Because enabled instance extensions are likewise not queryable, borrowed
instances are compute-only until the import contract carries an explicit
surface-extension attestation.

Volk's default dispatch mode stores instance and device entry points in
process-global tables. The Vulkan backend therefore permits one live backend
`Device` at a time and device enumeration/instance initialization must not run
while it is live. Keep the Vulkan loader module resident for process lifetime,
and reload the appropriate instance table whenever switching between a
borrowed instance and the process-owned instance.
Pin loader identity independently of `DynamicModule` ownership: Volk's default
`volkInitialize()` path owns its module internally, so an empty client-side
module handle does not mean that Volk is uninitialized. The first Vulkan
backend operation fixes either the default loader or a custom identity made of
the normalized absolute search directory plus exact library name. Reuse only
that identity and fail closed on default/custom, path, or name mismatches;
never replace `vkGetInstanceProcAddr` underneath existing instance state.
Device enumeration requests the default loader, so enumerating before a
custom-loader Device intentionally makes the later custom request invalid.
Client code linked to a separate Volk copy must initialize that copy through
`VulkanDeviceConfigExt::init_volk(handler)`, then call
`volkLoadInstanceOnly(instance)` from `readback_vulkan_device()` before using
`volkLoadDeviceTable()`. `volkInitialize()` or `volkInitializeCustom()` alone
loads only loader-level entry points and leaves `vkGetDeviceProcAddr` unset.

Native interop allocations must derive API creation fields and runtime wrapper
metadata from one plan. In particular, image type/dimension, extent, mip count,
format capabilities, and simultaneous-access policy must not diverge between
the native image and the wrapper returned to LuisaCompute.

Vulkan cannot query an imported image's current layout. A new native-image
identity therefore starts with every mip tracked as `VK_IMAGE_LAYOUT_UNDEFINED`;
the importer must publish the real external layout/access state through
`VulkanDeviceConfigExt::before_states()` before preserving or consuming
existing contents. Multiple wrappers created by the same backend device for
the same `VkImage` share one per-mip layout state, so aliases cannot publish
contradictory native metadata.

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

## CMake Patterns

```cmake
function(luisa_compute_add_backend name)
    cmake_parse_arguments(BACKEND "" "SUPPORT_DIR;BUILTIN_DIR" "SOURCES" ${ARGN})
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(_LUISA_BACKEND_LIBRARY_TYPE STATIC)
    else ()
        set(_LUISA_BACKEND_LIBRARY_TYPE MODULE)
    endif ()
    add_library(luisa-compute-backend-${name}
        ${_LUISA_BACKEND_LIBRARY_TYPE} ${BACKEND_SOURCES})
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

1. **Backend Boundary** — Desktop backends are shared plugins; iOS statically links the selected backend but keeps the same `DeviceInterface` boundary
2. **Handle-based Resources** — Opaque `uint64_t` handles for ABI stability
3. **Visitor Pattern** — Double dispatch via `MutableCommandVisitor` for type-safe command handling
4. **Version Checking** — Strict match between runtime and backend prevents ABI mismatches
5. **Validation Layer** — Optional wrap-around validation at device creation (or via `LUISA_ENABLE_VALIDATION`)
6. **BinaryIO** — Pluggable shader caching via `default_binary_io`
