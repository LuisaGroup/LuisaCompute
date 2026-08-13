---
name: lc_runtime
description: Runtime API: Context, Device, Stream, buffers, images, ray tracing, and rasterization.
---

# LuisaCompute Runtime API

Covers `luisa/runtime/` classes for GPU compute: context/device management, memory, execution, ray tracing, rasterization, presentation.

## Context

```cpp
#include <luisa/runtime/context.h>
luisa::compute::Context ctx{argv[0]};
// or: Context ctx{argv[0], data_dir};

for (auto &&backend : ctx.installed_backends()) {
    auto names = ctx.backend_device_names(backend);
}
Device device = ctx.create_default_device();
```

## Device

```cpp
#include <luisa/runtime/device.h>
Device device = ctx.create_device("cuda");  // or "dx", "metal", "vk", "hip", "fallback"

DeviceConfig cfg{.device_index = 0, .inqueue_buffer_limit = false};
Device device = ctx.create_device("cuda", &cfg, true/*validation*/);

auto backend = device.backend_name();
auto warp = device.compute_warp_size();
```

### Resource Creation
```cpp
Buffer<float> buf = device.create_buffer<float>(1024);
Buffer<MyStruct> sbuf = device.create_buffer<MyStruct>(100);
Image<float> img = device.create_image<float>(PixelStorage::FLOAT4, w, h);
Image<float> mip = device.create_image<float>(PixelStorage::FLOAT4, w, h, mips);
Volume<float> vol = device.create_volume<float>(PixelStorage::FLOAT4, w, h, d);
ByteBuffer bb = device.create_byte_buffer(size_bytes);
BindlessArray heap = device.create_bindless_array(65536);
IndirectDispatchBuffer indirect = device.create_indirect_dispatch_buffer(capacity);
```

## Stream

```cpp
#include <luisa/runtime/stream.h>
Stream stream = device.create_stream();
Stream compute = device.create_stream(StreamTag::COMPUTE);
Stream graphics = device.create_stream(StreamTag::GRAPHICS);
stream.set_name("my stream");
```

## Events

```cpp
#include <luisa/runtime/event.h>
Event event = device.create_event();
TimelineEvent timeline = device.create_timeline_event();

stream << event.signal();
stream << event.wait();
stream << graphics_event.wait(frame_index);
stream << graphics_event.signal(frame_index);
timeline.synchronize(frame_index);
```

## Buffer

```cpp
#include <luisa/runtime/buffer.h>
Buffer<float> buf = device.create_buffer<float>(1024);

// Transfer
stream << buf.copy_from(host_data);
stream << buf.copy_to(host_data);

// Views
auto view = buf.view(offset, count);
auto elem_view = buf.view().as<float>();  // for atomic operations
buf.set_name("vertex data");
```

#### Buffer-to-Buffer Copy

Use `BufferView::copy_from(BufferView<T>)` or `BufferView::copy_to(BufferView<T>)` — **these work in both normal and SAFE builds**.

```cpp
Buffer<float> src = device.create_buffer<float>(1024);
Buffer<float> dst = device.create_buffer<float>(1024);
Buffer<float> readback = device.create_buffer<float>(1024);

// ✅ Correct (SAFE-mode compatible): go through .view()
stream << dst.view().copy_from(src);            // BufferView::copy_from(BufferView)
stream << readback.view().copy_from(src);       // BufferView::copy_from(BufferView)

// ❌ Wrong (fails in SAFE mode): Buffer::copy_from(BufferView<T>)
// is guarded by #ifndef LUISA_ENABLE_SAFE_MODE
// stream << dst.copy_from(src.view());          // compile error in SAFE
```

#### SAFE Build Mode (`LUISA_ENABLE_SAFE_MODE`)

Define `LUISA_ENABLE_SAFE_MODE` at build time to **disable unsafe raw-pointer overloads**, enabling runtime validation of buffer creation. This is controlled by the cmake option `ENABLE_SAFE_MODE` in the project.

**What is excluded in SAFE mode** (`#ifndef LUISA_ENABLE_SAFE_MODE` blocks in `include/luisa/runtime/buffer.h`):

| Class | Excluded overloads |
|---|---|
| `Buffer<T>` | `copy_to(void*)`
`copy_to(BufferView<T>)`
`copy_to(const ByteBufferView&)`
`copy_from(const void*)`
`copy_from(const void*, move_only_function)`
`copy_from(BufferView<T>)`
`copy_from(const ByteBufferView&)` |
| `BufferView<T>` | `copy_to(void*)`
`copy_from(const void*)` |

**What remains available** (works in both modes):

| API | Example |
|---|---|
| `Buffer::copy_to(luisa::span<U>)` / `Buffer::copy_from(luisa::span<U>)` | `buf.copy_from(luisa::span{host_vec})` |
| `BufferView::copy_to(luisa::span<U>)` / `BufferView::copy_from(luisa::span<U>)` | `buf.view().copy_to(luisa::span{host_vec})` |
| `BufferView::copy_to(BufferView<T>)` / `BufferView::copy_from(BufferView<T>)` | `dst.view().copy_from(src)` |
| `BufferView::copy_to(const ByteBufferView&)` / `BufferView::copy_from(const ByteBufferView&)` | `buf.view().copy_to(byte_view)` |

**To pass the build in SAFE mode**: Always go through `BufferView` or `luisa::span` overloads instead of the `Buffer<T>` convenience overloads that are guarded. For buffer-to-buffer copy, change `dst.copy_from(src.view())` → `dst.view().copy_from(src)`. For raw-pointer transfers, change `buf.copy_from(data_ptr)` → `buf.copy_from(luisa::span{ptr, count})`.

## Image & Volume

```cpp
#include <luisa/runtime/image.h>
Image<float> img = device.create_image<float>(PixelStorage::FLOAT4, w, h);
Image<float> img2 = device.create_image<float>(swapchain.backend_storage(), size);
// Mipmapped: device.create_image<float>(PixelStorage::FLOAT4, w, h, mips);
// Simultaneous access: device.create_image<float>(PixelStorage::FLOAT4, w, h, 1, true);

#include <luisa/runtime/volume.h>
Volume<float> vol = device.create_volume<float>(PixelStorage::FLOAT4, w, h, d);
```

### Sparse Images and Volumes

Sparse image/volume mip counts follow the same convention as regular textures:
zero requests the full chain and larger requests are clamped to the logical
maximum. Tile map and unmap regions are validated against the selected mip's
ceil-divided tile grid, not the base extent or a floor-divided grid. Counts
must be nonzero and range arithmetic must not wrap. Sparse copy regions use
the same validation, convert tiles to texel offsets, and clip the final partial
tile to the selected mip extent; buffer-backed copies must provide enough
bytes for that clipped texel region.

Sparse buffers use the same nonzero-count and checked-range rules over a
ceil-divided byte tile grid. Every sparse map operation requires a valid heap
created by the same `DeviceInterface` as the sparse resource.

### Image in Kernels
```cpp
Kernel2D k = [&](ImageFloat img) {
    UInt2 coord = dispatch_id().xy();
    Float4 c = img.read(coord);
    img.write(coord, make_float4(1,0,0,1));
};
```

## BindlessArray

```cpp
#include <luisa/runtime/bindless_array.h>
BindlessArray heap = device.create_bindless_array(64);
heap.emplace_on_update(slot, buffer);
heap.emplace_on_update(slot, image, TextureSampler::linear_linear_mirror());
stream << heap.update() << synchronize();

// Kernel:
Kernel1D k = [&](Var<BindlessArray> heap) {
    auto v = heap.buffer<float>(slot).read(idx);
    auto c = heap.texture2d(slot).sample(uv);
};
```

## Swapchain

```cpp
#include <luisa/runtime/swapchain.h>
Swapchain swapchain = device.create_swapchain(stream, SwapchainOption{
    .display = window.native_display(),
    .window = window.native_handle(),
    .size = resolution,
    .wants_hdr = false,
    .wants_vsync = true,
    .back_buffer_count = 3});
stream << swapchain.present(image);
```

## Ray Tracing

```cpp
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/curve.h>

Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
Accel accel = device.create_accel();
accel.emplace_back(mesh, transform);
accel.emplace_back(mesh, transform, visibility_mask);
stream << mesh.build() << accel.build();
stream << accel.update_instance_buffer();

Curve curve = device.create_curve(CurveBasis::CUBIC_BSPLINE, cp_buf, seg_buf);
```

### Ray Tracing Kernel
```cpp
Kernel2D trace = [&](AccelVar accel, BufferFloat4 img) {
    Var<Ray> ray = make_ray(origin, direction);
    Var<TriangleHit> hit = accel.intersect(ray, {});
    $if (!hit->miss()) {
        Float3 c = triangle_interpolate(hit.bary, v0, v1, v2);
    };
};
```

## Rasterization

```cpp
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_shader.h>

DepthBuffer depth = device.create_depth_buffer(DepthFormat::D32, size);
auto raster_shader = device.compile(raster_kernel, mesh_format);
RasterScene scene = device.create_raster_scene(vertex_buffer, index_buffer);
```

## CommandList

Batch commands for efficient submission:
```cpp
CommandList cmdlist = CommandList::create();
cmdlist << kernel.dispatch(w, h) << buffer.copy_to(host_data);
stream << cmdlist.commit() << synchronize();
```

> Prefer merging dispatch + transfers into one `CommandList` + single commit/synchronize over separate stream submissions.

## Complete Example

```cpp
#include <luisa/luisa-compute.h>
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context ctx{argv[0]};
    Device device = ctx.create_device("cuda");
    Stream stream = device.create_stream();
    Buffer<float> buf = device.create_buffer<float>(1024);

    Kernel1D k = [&](BufferVar<float> buf) {
        auto idx = dispatch_id().x;
        buf.write(idx, buf.read(idx) + 1.0f);
    };

    auto shader = device.compile(k);
    stream << shader(buf).dispatch(1024) << synchronize();
}
```

## Common Patterns

### Multi-Stream Sync
```cpp
Stream compute = device.create_stream(StreamTag::COMPUTE);
Stream graphics = device.create_stream(StreamTag::GRAPHICS);
Event event = device.create_event();
compute << shader().dispatch(w, h) << event.signal();
graphics << event.wait() << swapchain.present(img);
```

### Triple Buffering
```cpp
TimelineEvent timeline = device.create_timeline_event();
uint64_t frame = 0;
while (running) {
    if (frame >= 3) timeline.synchronize(frame - 2);
    stream << shader().dispatch(w, h) << timeline.signal(++frame);
}
```

### Buffer Upload/Download

Always prefer `luisa::span<T>` overloads for SAFE-mode compatibility:

```cpp
luisa::vector<float> host_data(1024, 1.0f);

// ✅ span-based (SAFE-mode compatible)
stream << buf.copy_from(luisa::span{host_data}) << synchronize();
stream << buf.copy_to(luisa::span{host_data}) << synchronize();

// ❌ raw-pointer (fails in SAFE mode)
// stream << buf.copy_to(host_data.data()) << synchronize();
// stream << buf.copy_from(host_data.data()) << synchronize();
```

## Key Headers

| Header | Class |
|---|---|
| `luisa/runtime/context.h` | Context |
| `luisa/runtime/device.h` | Device |
| `luisa/runtime/stream.h` | Stream |
| `luisa/runtime/event.h` | Event, TimelineEvent |
| `luisa/runtime/buffer.h` | Buffer |
| `luisa/runtime/image.h` | Image |
| `luisa/runtime/volume.h` | Volume |
| `luisa/runtime/swapchain.h` | Swapchain |
| `luisa/runtime/bindless_array.h` | BindlessArray |
| `luisa/runtime/dispatch_buffer.h` | IndirectDispatchBuffer |
| `luisa/runtime/command_list.h` | CommandList |
| `luisa/runtime/rtx/accel.h` | Accel |
| `luisa/runtime/rtx/mesh.h` | Mesh |
| `luisa/runtime/rtx/curve.h` | Curve |
| `luisa/runtime/rtx/ray.h` | Ray, hit types |
| `luisa/runtime/raster/raster_shader.h` | RasterShader |
| `luisa/runtime/raster/raster_scene.h` | RasterScene |
