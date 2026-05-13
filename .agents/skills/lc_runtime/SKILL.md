---
name: lc_runtime
---

# LuisaCompute Runtime API Skill

This skill documents the usage of `luisa/runtime/` classes in LuisaCompute, a high-performance compute framework for GPU acceleration.

## Overview

The LuisaCompute runtime API provides classes for:
- **Context & Device Management**: Initialize compute context and create backend devices
- **Memory Management**: Buffers, Images, Volumes, and specialized memory resources
- **Execution**: Streams, Events, Synchronization
- **Ray Tracing**: Acceleration structures, Meshes, Curves
- **Rasterization**: Raster shaders and scenes
- **Presentation**: Swapchains for window rendering

## Core Classes Usage

### Context

Entry point for the runtime. Manages backend discovery and device creation.

```cpp
#include <luisa/runtime/context.h>

// Initialize with program path (typically argv[0])
luisa::compute::Context context{argv[0]};

// Alternative: with data directory
Context context{argv[0], data_dir};

// Query installed backends
for (auto &&backend : context.installed_backends()) {
    auto device_names = context.backend_device_names(backend);
}

// Create default device
Device device = context.create_default_device();
```

### Device

Primary interface for creating GPU resources and compiling kernels.

```cpp
#include <luisa/runtime/device.h>

// Create device with specific backend
Device device = context.create_device("cuda");  // or "dx", "cpu", "metal"

// Create with configuration
DeviceConfig config{
    .device_index = 0,
    .inqueue_buffer_limit = false};
Device device = context.create_device("cuda", &config, true /*enable_validation*/);

// Query device properties
auto backend = device.backend_name();
auto warp_size = device.compute_warp_size();
```

#### Creating Resources

```cpp
// Buffers
Buffer<float> buffer = device.create_buffer<float>(1024);
Buffer<MyStruct> struct_buffer = device.create_buffer<MyStruct>(100);

// Images
Image<float> image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
Image<float> image_with_mips = device.create_image<float>(PixelStorage::FLOAT4, width, height, mip_levels);

// Volumes (3D textures)
Volume<float> volume = device.create_volume<float>(PixelStorage::FLOAT4, width, height, depth);

// Byte buffers (untyped)
ByteBuffer byte_buffer = device.create_byte_buffer(size_in_bytes);

// Bindless arrays
BindlessArray heap = device.create_bindless_array(65536);

// Indirect dispatch buffer
IndirectDispatchBuffer indirect = device.create_indirect_dispatch_buffer(capacity);
```

### Stream

Command submission and execution queue.

```cpp
#include <luisa/runtime/stream.h>

// Create compute stream (default)
Stream stream = device.create_stream();
Stream compute_stream = device.create_stream(StreamTag::COMPUTE);

// Create graphics stream (for presentation)
Stream graphics_stream = device.create_stream(StreamTag::GRAPHICS);

// Named streams for debugging
stream.set_name("my compute stream");
```

### Event & TimelineEvent

Synchronization primitives between streams and host.

```cpp
#include <luisa/runtime/event.h>

// Binary event (signal/wait)
Event event = device.create_event();

// Timeline event (for frame pacing)
TimelineEvent timeline = device.create_timeline_event();

// Usage in command lists:
stream << compute_event.signal();
stream << compute_event.wait();
stream << graphics_event.wait(frame_index);
stream << graphics_event.signal(frame_index);

// Host synchronization
timeline.synchronize(frame_index);
```

### Buffer

Typed GPU memory buffer.

```cpp
#include <luisa/runtime/buffer.h>

// Creation
Buffer<float> buffer = device.create_buffer<float>(1024);

// Data transfer (via stream)
stream << buffer.copy_from(host_data);
stream << buffer.copy_to(host_data);

// Views
auto view = buffer.view(offset, count);
auto element_view = buffer.view().as<float>();  // for atomic operations

// Named buffers for debugging
buffer.set_name("vertex data");
```

#### Buffer in Kernels

```cpp
Buffer<float> buffer = device.create_buffer<float>(1024);

Kernel1D kernel = [&](BufferVar<float> buf) {
    // Read/write
    float value = buf.read(index);
    buf.write(index, value + 1.0f);
    
    // Atomic operations
    buf.atomic(index).fetch_add(1.0f);
    
    // Volatile operations
    float v = buf.volatile_read(index);
    buf.volatile_write(index, v);
};
```

### Image

2D GPU texture/image.

```cpp
#include <luisa/runtime/image.h>

// Creation
Image<float> image = device.create_image<float>(PixelStorage::FLOAT4, width, height);
Image<float> storage_image = device.create_image<float>(swapchain.backend_storage(), size);

// Mipmapped image
Image<float> image = device.create_image<float>(PixelStorage::FLOAT4, width, height, mip_levels);

// Simultaneous access (for multi-stream usage)
Image<float> image = device.create_image<float>(PixelStorage::FLOAT4, width, height, 1, true);
```

#### Image in Kernels

```cpp
Kernel2D kernel = [&](ImageFloat image) {
    UInt2 coord = dispatch_id().xy();
    
    // Read
    Float4 color = image.read(coord);
    
    // Write
    image.write(coord, make_float4(1.0f, 0.0f, 0.0f, 1.0f));
};
```

### Volume

3D GPU texture/volume.

```cpp
#include <luisa/runtime/volume.h>

Volume<float> volume = device.create_volume<float>(PixelStorage::FLOAT4, width, height, depth);
```

### BindlessArray

Bindless resource array for dynamic indexing.

```cpp
#include <luisa/runtime/bindless_array.h>

BindlessArray heap = device.create_bindless_array(64);

// Emplace resources
heap.emplace_on_update(slot_index, buffer);
heap.emplace_on_update(slot_index, image, TextureSampler::linear_linear_mirror());

// Update to apply changes
stream << heap.update() << synchronize();
```

#### Bindless in Kernels

```cpp
Kernel1D kernel = [&](Var<BindlessArray> heap) {
    // Access buffer
    auto value = heap.buffer<float>(slot).read(index);
    
    // Access texture
    auto color = heap.texture2d(slot).sample(uv);
};
```

### Swapchain

Window presentation.

```cpp
#include <luisa/runtime/swapchain.h>

// Creation
Swapchain swapchain = device.create_swapchain(
    stream,
    SwapchainOption{
        .display = window.native_display(),
        .window = window.native_handle(),
        .size = resolution,
        .wants_hdr = false,
        .wants_vsync = true,
        .back_buffer_count = 3});

// Presentation
stream << swapchain.present(image);
```

## Ray Tracing Classes

### Accel (Acceleration Structure)

```cpp
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>

// Create mesh
Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);

// Create acceleration structure
Accel accel = device.create_accel();

// Add instances
accel.emplace_back(mesh, transform_matrix);
accel.emplace_back(mesh, transform_matrix, visibility_mask);

// Build/update
stream << mesh.build()
       << accel.build();

// Update instance transforms
stream << accel.update_instance_buffer();
```

#### Ray Tracing in Kernels

```cpp
Kernel2D trace_kernel = [&](AccelVar accel, BufferFloat4 image) {
    Var<Ray> ray = make_ray(origin, direction);
    Var<TriangleHit> hit = accel.intersect(ray, {});
    
    $if (!hit->miss()) {
        Float3 color = triangle_interpolate(hit.bary, v0, v1, v2);
    };
};
```

### Curve

```cpp
#include <luisa/runtime/rtx/curve.h>

Curve curve = device.create_curve(
    CurveBasis::CUBIC_BSPLINE,
    control_point_buffer,
    segment_buffer);
```

## Rasterization Classes

### DepthBuffer

```cpp
#include <luisa/runtime/raster/depth_buffer.h>

DepthBuffer depth = device.create_depth_buffer(DepthFormat::D32, size);
```

### RasterScene

```cpp
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_shader.h>

// Compile raster shader
auto raster_shader = device.compile(raster_kernel, mesh_format);

// Create raster scene
RasterScene scene = device.create_raster_scene(vertex_buffer, index_buffer);
```

## Command List

Batch commands for efficient submission:

```cpp
#include <luisa/runtime/command_list.h>

CommandList cmdlist = CommandList::create();
cmdlist << kernel.dispatch(width, height)
        << buffer.copy_to(host_data);
        
stream << cmdlist.commit() << synchronize();
```

## Complete Example

```cpp
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    // Initialize
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    
    // Create resources
    Buffer<float> buffer = device.create_buffer<float>(1024);
    
    // Define kernel
    Kernel1D kernel = [&](BufferVar<float> buf) {
        auto idx = dispatch_id().x;
        buf.write(idx, buf.read(idx) + 1.0f);
    };
    
    // Compile and execute
    auto shader = device.compile(kernel);
    stream << shader(buffer).dispatch(1024)
           << synchronize();
    
    return 0;
}
```

## Common Patterns

### Multi-Stream Synchronization

```cpp
Stream compute = device.create_stream(StreamTag::COMPUTE);
Stream graphics = device.create_stream(StreamTag::GRAPHICS);
Event event = device.create_event();

compute << shader().dispatch(width, height)
        << event.signal();
        
graphics << event.wait()
         << swapchain.present(image);
```

### Triple Buffering with Timeline Events

```cpp
static constexpr uint32_t frame_count = 3;
TimelineEvent timeline = device.create_timeline_event();
uint64_t frame_index = 0;

while (running) {
    // Wait for frame N-3 to complete
    if (frame_index >= frame_count) {
        timeline.synchronize(frame_index - (frame_count - 1));
    }
    
    // Submit work
    stream << shader().dispatch(width, height)
           << timeline.signal(++frame_index);
}
```

### Buffer Upload/Download

```cpp
// Upload
stream << buffer.copy_from(host_data) << synchronize();

// Download
stream << buffer.copy_to(host_data) << synchronize();

// Or use luisa::vector for automatic management
luisa::vector<float> data(size);
stream << buffer.copy_to(data.data()) << synchronize();
```

### Merge Commit and Synchronize

`commit()` and `synchronize()` are performance-costly operations. Avoid splitting a single logical workload into multiple stream submissions when possible.

Merge the download (or upload) commands into the same `CommandList` before committing, so both kernel dispatches and data transfers are submitted together and synchronized once:

```cpp
CommandList cmdlist = CommandList::create();
cmdlist << kernel.dispatch(width, height)
        << buffer.copy_to(host_data);

// Single commit + synchronize instead of separate submissions
stream << cmdlist.commit() << synchronize();
```

> **Merge-commit change:** Uncommitted changes in `examples/compute/tokenize/bm25_scorer.cpp` and `examples/compute/tokenize/simhash.cpp` adopted this pattern by moving `copy_to` calls into the `CommandList` and combining them with `.commit() << synchronize()` into a single stream operation.

## File List

Key runtime headers:
- `luisa/runtime/context.h` - Context class
- `luisa/runtime/device.h` - Device class
- `luisa/runtime/stream.h` - Stream class
- `luisa/runtime/event.h` - Event and TimelineEvent
- `luisa/runtime/buffer.h` - Buffer class
- `luisa/runtime/image.h` - Image class
- `luisa/runtime/volume.h` - Volume class
- `luisa/runtime/swapchain.h` - Swapchain class
- `luisa/runtime/bindless_array.h` - BindlessArray class
- `luisa/runtime/dispatch_buffer.h` - IndirectDispatchBuffer
- `luisa/runtime/command_list.h` - CommandList class
- `luisa/runtime/rtx/accel.h` - Accel (acceleration structure)
- `luisa/runtime/rtx/mesh.h` - Mesh class
- `luisa/runtime/rtx/curve.h` - Curve class
- `luisa/runtime/rtx/ray.h` - Ray and hit types
- `luisa/runtime/raster/raster_shader.h` - RasterShader
- `luisa/runtime/raster/raster_scene.h` - RasterScene
