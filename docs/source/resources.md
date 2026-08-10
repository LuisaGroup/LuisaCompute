# Resources and Runtime

This document covers the LuisaCompute runtime system, including resource management, command execution, and synchronization.

## Overview

The LuisaCompute runtime provides a unified abstraction layer over various GPU APIs (CUDA, DirectX, Metal). It handles:

- **Resource Management**: Buffers, images, volumes, and acceleration structures
- **Command Scheduling**: Asynchronous execution with automatic dependency tracking
- **Memory Transfers**: Efficient data movement between host and device
- **Synchronization**: Stream and event-based coordination

## Context

The `Context` is the entry point to the LuisaCompute runtime. It manages backend plugins and global configuration.

### Creating a Context

```cpp
#include <luisa/luisa-compute.h>

// From command-line arguments
int main(int argc, char* argv[]) {
    Context context{argv[0]};
    // ...
}

// From explicit path
Context context{"/path/to/executable"};

// With custom data directory (for caches)
Context context{"/path/to/executable", "/path/to/data"};
```

### Querying Available Backends

```cpp
Context context{argv[0]};

// List installed backends
auto backends = context.installed_backends();
for (auto& name : backends) {
    std::cout << "Backend: " << name << std::endl;
}

// Query device names for a backend
auto cuda_devices = context.backend_device_names("cuda");
for (auto& name : cuda_devices) {
    std::cout << "CUDA device: " << name << std::endl;
}
```

## Device

A `Device` represents a physical or virtual GPU/CPU for computation.

### Creating Devices

```cpp
// Create specific backend
Device cuda = context.create_device("cuda");
Device dx = context.create_device("dx");           // Windows
Device metal = context.create_device("metal");     // macOS
Device fallback = context.create_device("fallback"); // Requires LLVM + Embree

// Create with validation enabled
Device device = context.create_device("cuda", nullptr, true);

// Or via environment variable
// LUISA_ENABLE_VALIDATION=1 ./program

// Use first available backend
Device device = context.create_default_device();
```

### Device Properties

```cpp
// Query device information
std::string backend = device.backend_name();       // "cuda", "dx", etc.
void* native_handle = device.native_handle();      // CUcontext, ID3D12Device*, etc.
uint warp_size = device.compute_warp_size();       // SIMD width (e.g., 32 for CUDA)
size_t granularity = device.memory_granularity();  // Allocation alignment

// Check if device is valid
if (device) {
    // Device is properly initialized
}
```

### Backend Extensions

Devices may support backend-specific extensions:

```cpp
// CUDA-specific extensions
if (auto cuda_ext = device.extension<CUDAExternalExt>()) {
    // Use CUDA interop features
}

// DirectX-specific extensions
if (auto dx_ext = device.extension<DXConfigExt>()) {
    // Use DirectX-specific features
}
```

## Buffers

Buffers are linear memory regions for structured data storage.

### Creating Buffers

```cpp
// Simple buffer of 1000 floats
Buffer<float> float_buf = device.create_buffer<float>(1000);

// Buffer of vectors
Buffer<float4> vec4_buf = device.create_buffer<float4>(1024);

// Buffer of custom structures (must be reflected with LUISA_STRUCT)
struct Particle {
    float3 position;
    float3 velocity;
    float mass;
};
LUISA_STRUCT(Particle, position, velocity, mass) {};

Buffer<Particle> particles = device.create_buffer<Particle>(10000);
```

### Buffer Operations

```cpp
// Upload from host
std::vector<float> host_data(1000, 1.0f);
stream << float_buf.copy_from(host_data.data());

// Download to host
std::vector<float> result(1000);
stream << float_buf.copy_to(result.data());

// Copy between buffers
Buffer<float> other_buf = device.create_buffer<float>(1000);
stream << other_buf.copy_from(float_buf);

// Fill with value (in a kernel)
Kernel1D fill = [&](BufferFloat buf, Float value) noexcept {
    Var i = dispatch_id().x;
    buf.write(i, value);
};
auto fill_shader = device.compile(fill);
stream << fill_shader(float_buf, 3.14f).dispatch(1000);
```

### Sub-Buffers

```cpp
// Create a view into a portion of a buffer
Buffer<float> large_buf = device.create_buffer<float>(10000);

// View elements [1000, 5999]
BufferView<float> sub = large_buf.view(1000, 5000);

// Use sub-buffer in kernels
stream << kernel(sub).dispatch(5000);

// Copy to/from sub-buffers
stream << sub.copy_from(host_data.data());
```

### Byte Buffers

For untyped/raw memory access:

```cpp
// Create byte buffer (1000 bytes)
ByteBuffer raw = device.create_byte_buffer(1000);

// Use in kernels with manual pointer arithmetic
Kernel1D process = [&](ByteBuffer raw) noexcept {
    Var idx = dispatch_id().x;
    // Access via buffer pointer...
};
```

## Images

Images are 2D textures with hardware-accelerated sampling and caching.

### Creating Images

```cpp
// Basic RGBA8 image (1024x1024)
// Template type determines how pixels are read/written
// Storage format determines internal memory layout
Image<float> image = device.create_image<float>(
    PixelStorage::BYTE4,    // 4 channels, 8 bits each
    1024, 1024              // Width, height
);

// Floating point format
Image<float> hdr_image = device.create_image<float>(
    PixelStorage::FLOAT4,   // 4 channels, 32-bit float each
    1920, 1080
);

// With mipmaps
Image<float> mip_image = device.create_image<float>(
    PixelStorage::BYTE4,
    1024, 1024,
    10                      // 10 mipmap levels (1024 -> 1)
);

// Simultaneous access (for concurrent read/write)
Image<float> shared = device.create_image<float>(
    PixelStorage::BYTE4,
    1024, 1024, 1,          // 1 mipmap level
    true                    // Simultaneous access
);
```

### Pixel Storage Formats

| Storage Format | Channels | Bits per Channel | C++ Read/Write Type |
|---------------|----------|------------------|---------------------|
| `BYTE1` | 1 | 8 | `float` |
| `BYTE2` | 2 | 8 | `float2` |
| `BYTE4` | 4 | 8 | `float4` |
| `SHORT1` | 1 | 16 | `float` |
| `SHORT2` | 2 | 16 | `float2` |
| `SHORT4` | 4 | 16 | `float4` |
| `FLOAT1` | 1 | 32 | `float` |
| `FLOAT2` | 2 | 32 | `float2` |
| `FLOAT4` | 4 | 32 | `float4` |

### Image Operations

```cpp
// Upload from host
std::vector<std::byte> pixels(width * height * 4);
// ... fill pixels ...
stream << image.copy_from(pixels.data());

// Download to host
stream << image.copy_to(pixels.data());

// Copy between images
Image<float> dest = device.create_image<float>(PixelStorage::BYTE4, 1024, 1024);
stream << dest.copy_from(image);

// Generate mipmaps
stream << image.generate_mipmaps();
```

### Image Views

```cpp
// View a specific mipmap level
ImageView<float> level0 = image.view(0);  // Full resolution
ImageView<float> level1 = image.view(1);  // Half resolution

// View in kernel
Kernel2D process = [&](ImageFloat image) noexcept {
    Var coord = dispatch_id().xy();
    Float4 color = image.read(coord);
    image.write(coord, color * 0.5f);
};
stream << process(image.view(0)).dispatch(1024, 1024);
```

## Volumes

Volumes are 3D textures, useful for volumetric data and 3D lookups.

```cpp
// Create 256^3 volume
Volume<float> volume = device.create_volume<float>(
    PixelStorage::BYTE4,
    256, 256, 256
);

// Upload 3D data
std::vector<std::byte> voxel_data(256 * 256 * 256 * 4);
stream << volume.copy_from(voxel_data.data());

// Use in kernel
Kernel3D process = [&](VolumeFloat vol) noexcept {
    Var coord = dispatch_id().xyz();
    Float4 value = vol.read(coord);
    vol.write(coord, value * 2.0f);
};
stream << process(volume).dispatch(256, 256, 256);
```

## Bindless Arrays

Bindless arrays allow dynamic indexing of resources in shaders, bypassing binding limits.

### Creating Bindless Arrays

```cpp
// Create with 65536 slots (default)
BindlessArray bindless = device.create_bindless_array();

// Or specify slot count
BindlessArray bindless = device.create_bindless_array(1024);

// Specify allowed resource types
BindlessArray textures_only = device.create_bindless_array(
    1024,
    BindlessSlotType::TEXTURE_ONLY  // Only images/volumes
);
```

### Using Bindless Arrays

```cpp
// Prepare resources
std::vector<Image<float>> textures;
for (int i = 0; i < 100; i++) {
    textures.push_back(device.create_image<float>(PixelStorage::BYTE4, 512, 512));
}

// Bind resources to slots
luisa::vector<BindlessArrayUpdate> updates;
for (int i = 0; i < textures.size(); i++) {
    updates.emplace_back(BindlessArrayUpdate{
        BindlessArrayUpdate::Operation::Emplace,
        i,                      // Slot index
        textures[i].handle(),   // Resource handle
        Sampler::linear_linear_mirror()  // Sampler
    });
}

// Apply updates
stream << bindless.update(updates);

// Use in kernel
Kernel2D sample_textures = [&](BindlessArray textures, Int tex_id) noexcept {
    Var coord = dispatch_id().xy();
    Float2 uv = make_float2(coord) / 512.0f;
    
    // Sample from dynamically selected texture
    Float4 color = textures.sample(tex_id, uv);
    // ...
};
```

## Acceleration Structures (Ray Tracing)

For hardware-accelerated ray-scene intersection.

### Meshes

```cpp
// Create mesh from vertex and index buffers
Buffer<float3> vertices = device.create_buffer<float3>(num_vertices);
Buffer<Triangle> triangles = device.create_buffer<Triangle>(num_triangles);
// ... fill buffers ...

Mesh mesh = device.create_mesh(vertices, triangles);

// Build acceleration structure
stream << mesh.build();
```

### Acceleration Structure (Accel)

```cpp
// Create top-level acceleration structure
Accel accel = device.create_accel();

// Add meshes to the accel
luisa::vector<Accel::Modification> mods;
mods.push_back(Accel::Modification{
    Accel::Modification::operation_set_mesh,
    mesh.handle(),      // Bottom-level mesh
    float4x4::identity(), // Transform matrix
    0,                  // Instance ID (visible in shader)
    0xff,               // Visibility mask
    0,                  // Instance index
    Accel::Modification::flag_mesh
});

// Build/update accel
stream << accel.build(mods);

// Use in ray tracing kernel
Kernel2D trace = [&](ImageFloat image, Accel accel) noexcept {
    Var coord = dispatch_id().xy();
    
    Ray ray = ...;
    Var<SurfaceHit> hit = accel.intersect(ray, {});
    
    $if (!hit->miss()) {
        // Process hit
    };
};
```

## Streams

Streams are command queues for asynchronous execution.

### Creating Streams

```cpp
// Compute stream (default)
Stream compute = device.create_stream(StreamTag::COMPUTE);

// Graphics stream (for rasterization)
Stream graphics = device.create_stream(StreamTag::GRAPHICS);

// Copy/streaming stream
Stream transfer = device.create_stream(StreamTag::COPY);
```

### Submitting Commands

```cpp
// Simple command sequence
stream << buffer.copy_from(host_data)
       << shader.dispatch(1024, 1024)
       << buffer.copy_to(result_data)
       << synchronize();

// Complex command buffer
{
    auto cmd_list = stream.command_buffer();
    cmd_list << buffer_a.copy_from(data_a)
             << buffer_b.copy_from(data_b)
             << compute_shader(buffer_a, buffer_b)
                 .dispatch(1024, 1024)
             << buffer_c.copy_from(buffer_a)
             << commit();
}

// Host callbacks
stream << shader.dispatch(1024, 1024)
       << [&]() {
           // This runs on host after shader completes
           std::cout << "Shader finished!" << std::endl;
       }
       << synchronize();
```

### Stream Synchronization

```cpp
// Synchronize specific stream
stream.synchronize();
stream << synchronize();

// Wait for specific event
Event event = device.create_event();
stream << shader.dispatch(1024, 1024)
       << event.signal()
       << synchronize();

event.synchronize();  // Block until signaled
```

## Events

Events provide synchronization between streams or with the host.

### Basic Usage

```cpp
Event event = device.create_event();

// Stream A signals
Stream stream_a = device.create_stream();
stream_a << command_a
         << event.signal()
         << synchronize();

// Stream B waits
Stream stream_b = device.create_stream();
stream_b << event.wait()
         << command_b  // Executes after command_a completes
         << synchronize();
```

### Timeline Events

For more complex synchronization patterns:

```cpp
TimelineEvent timeline = device.create_timeline_event();

// Signal with value
stream_a << timeline.signal(100);

// Wait for specific value
stream_b << timeline.wait(100);  // Wait until value >= 100
```

## Memory Model and RAII

All resources follow RAII (Resource Acquisition Is Initialization):

```cpp
{
    // Resource created
    Buffer<float> buffer = device.create_buffer<float>(1000);
    Image<float> image = device.create_image<float>(PixelStorage::BYTE4, 1024, 1024);
    
    // Use resources...
    
} // Resources automatically released here
```

### Move Semantics

Resources are move-only (cannot be copied):

```cpp
Buffer<float> buf1 = device.create_buffer<float>(1000);
Buffer<float> buf2 = std::move(buf1);  // Valid
// Buffer<float> buf3 = buf2;          // Error: copy not allowed
```

### Resource Lifetimes

Important: Ensure resources outlive their commands:

```cpp
void problematic(Stream& stream) {
    Buffer<float> local = device.create_buffer<float>(1000);
    stream << kernel(local).dispatch(1000);
    // local destroyed here, but command may not have executed!
}

// Better
void correct(Stream& stream, Buffer<float>& buffer) {
    stream << kernel(buffer).dispatch(1000);
}  // Caller keeps buffer alive
```

## Command Scheduling

LuisaCompute automatically analyzes command dependencies and optimizes execution order:

```cpp
// These commands have no dependencies - may execute in parallel
stream << buf_a.copy_from(host_a)    // Write to buf_a
       << buf_b.copy_from(host_b)    // Write to buf_b (independent)
       << kernel(buf_a).dispatch(n)   // Read buf_a
       << kernel(buf_b).dispatch(n)   // Read buf_b (can run concurrently)
       << synchronize();

// Dependencies are automatically tracked
stream << buf.write(data)            // Write to buf
       << kernel(buf).dispatch(n)    // Read buf - waits for write
       << buf.copy_to(result)        // Read buf - waits for kernel
       << synchronize();
```

## Shaders

Shaders are compiled kernels ready for dispatch on the device. The `Shader<N, Args...>` type captures both the dimensionality (1D/2D/3D) and the argument types at compile time.

### Compiling Kernels

```cpp
// Define a kernel
Kernel2D my_kernel = [&](ImageFloat image, Float time) noexcept {
    Var coord = dispatch_id().xy();
    // ...
};

// Compile into a shader
auto shader = device.compile(my_kernel);
```

> **Note:** Compilation blocks the calling thread. For large kernels this can take a significant amount of time. You may compile multiple kernels concurrently using thread pools.

### Shader Options

```cpp
ShaderOption option{
    .enable_cache = true,         // Cache compiled shader on disk
    .enable_fast_math = true,     // Allow fast math optimizations
    .enable_debug_info = false,   // Include debug symbols
    .name = "my_shader"           // Name for profiling/debugging
};

auto shader = device.compile(my_kernel, option);
```

### Dispatching Shaders

Invoke `operator()` with arguments, then call `.dispatch()` with the grid size:

```cpp
// 1D dispatch
Shader1D<Buffer<float>, float> s1d = device.compile(kernel1d);
stream << s1d(buffer, 3.14f).dispatch(1024);

// 2D dispatch
Shader2D<Image<float>> s2d = device.compile(kernel2d);
stream << s2d(image).dispatch(width, height);
// or with uint2:
stream << s2d(image).dispatch(make_uint2(width, height));

// 3D dispatch
Shader3D<Volume<float>> s3d = device.compile(kernel3d);
stream << s3d(volume).dispatch(64, 64, 64);
```

### Indirect Dispatch

Dispatch sizes can be determined on the GPU via `IndirectDispatchBuffer`:

```cpp
// Create indirect dispatch buffer (capacity = max dispatches)
auto indirect = device.create_indirect_dispatch_buffer(16);

// A kernel writes dispatch parameters to the buffer
Kernel1D prepare = [&](IndirectDispatchBuffer buf) noexcept {
    // Set dispatch 0 to (256, 256, 1)
    buf->set_dispatch_count(1u);
    buf->set_kernel(dispatch_id().x, make_uint3(256u, 256u, 1u));
};

// Dispatch using indirect buffer
stream << prepare_shader(indirect).dispatch(1)
       << compute_shader(args...).dispatch(indirect);
```

### Shader Caching

Most backends cache compiled shaders at `<build-folder>/bin/.cache`. On subsequent runs, compilation is skipped if the kernel hasn't changed. You can control caching via `ShaderOption::enable_cache`.

### Shader Aliases

```cpp
template<typename... Args>
using Shader1D = Shader<1, Args...>;

template<typename... Args>
using Shader2D = Shader<2, Args...>;

template<typename... Args>
using Shader3D = Shader<3, Args...>;
```

## Swapchain

A `Swapchain` represents a presentable surface (window) for displaying rendered images.

### Creating a Swapchain

```cpp
auto swapchain = device.create_swapchain(
    stream,
    SwapchainOption{
        .display = window_native_display_handle,
        .window = window_native_handle,
        .size = make_uint2(width, height),
        .wants_hdr = false,
        .wants_vsync = true,
        .back_buffer_count = 2
    }
);
```

### Presenting Frames

Use the `.present()` method to display an image to the window:

```cpp
Image<float> framebuffer = device.create_image<float>(
    PixelStorage::BYTE4, width, height);

// Render loop
while (running) {
    stream << render_shader(framebuffer).dispatch(width, height)
           << swapchain.present(framebuffer);
}
```

### Querying Backend Storage

```cpp
// Get the pixel format the swapchain uses internally
PixelStorage storage = swapchain.backend_storage();
```

## Sampler

`Sampler` configures how textures are filtered and addressed when sampled in shaders. Samplers are used with bindless arrays to bind images/volumes with specific sampling behavior.

### Filter Modes

| Filter | Description |
|--------|-------------|
| `POINT` | Nearest-neighbor (no interpolation) |
| `LINEAR_POINT` | Bilinear filtering, point mipmap selection |
| `LINEAR_LINEAR` | Bilinear filtering, linear mipmap interpolation (trilinear) |
| `ANISOTROPIC` | Anisotropic filtering (highest quality) |

### Address Modes

| Address | Description |
|---------|-------------|
| `EDGE` | Clamp to edge pixels |
| `REPEAT` | Tile/wrap around |
| `MIRROR` | Mirror at boundaries |
| `ZERO` | Return zero outside [0, 1] |

### Construction

Samplers are lightweight value types. Use the constructor or convenience factory methods:

```cpp
// Constructor
Sampler s{Sampler::Filter::LINEAR_LINEAR, Sampler::Address::REPEAT};

// Factory methods (filter_address pattern)
auto s1 = Sampler::point_edge();
auto s2 = Sampler::linear_point_repeat();
auto s3 = Sampler::linear_linear_mirror();
auto s4 = Sampler::anisotropic_repeat();
```

All 16 combinations of 4 filters × 4 address modes have dedicated factory methods following the `filter_address()` naming pattern.

## Depth Buffer

`DepthBuffer` provides a depth/stencil buffer for rasterization.

```cpp
// Create depth buffer
auto depth = device.create_depth_buffer(DepthFormat::D32, make_uint2(width, height));

// Clear to a specific depth value
stream << depth.clear(1.0f);

// Query properties
uint2 size = depth.size();
DepthFormat fmt = depth.format();

// Convert to ImageView for reading in compute shaders
// (writing to this image in compute kernels is illegal)
ImageView<float> depth_img = depth.to_img();
```

### Depth Formats

| Format | Description |
|--------|-------------|
| `D16` | 16-bit depth |
| `D24S8` | 24-bit depth + 8-bit stencil |
| `D32` | 32-bit float depth |

## Raster Pipeline

LuisaCompute provides an experimental rasterization pipeline for rendering triangle meshes with programmable vertex and pixel shaders. The raster pipeline requires a **graphics stream** (`StreamTag::GRAPHICS`).

### Overview

The raster pipeline consists of:
- **`RasterShader<Args...>`** — Compiled vertex + pixel shader pair
- **`DepthBuffer`** — Depth/stencil attachment
- **`RasterState`** — Pipeline state (cull mode, fill mode, blend, etc.)
- **`MeshFormat`** — Vertex attribute layout description
- **`Viewport`** — Render target region

### Basic Raster Usage

```cpp
// Create a graphics stream
Stream graphics = device.create_stream(StreamTag::GRAPHICS);

// Create render targets
Image<float> color = device.create_image<float>(PixelStorage::BYTE4, width, height);
auto depth = device.create_depth_buffer(DepthFormat::D32, make_uint2(width, height));

// Compile a raster shader (vertex + pixel)
auto raster = device.compile_raster_shader(vertex_kernel, pixel_kernel, option);

// Draw
graphics << raster(args...)
    .draw(scene_meshes, mesh_format, viewport, raster_state, &depth, color);
```

> **Note:** The raster pipeline is not available on all backends. Check your backend's documentation for support status.

## Best Practices

### 1. Batch Commands

```cpp
// Good: Fewer, larger command batches
stream << cmd1 << cmd2 << cmd3 << cmd4 << synchronize();

// Avoid: Many small submissions
stream << cmd1 << synchronize();
stream << cmd2 << synchronize();
```

### 2. Reuse Resources

```cpp
// Good: Allocate once, reuse
Buffer<float> buffer = device.create_buffer<float>(max_size);
for (int frame = 0; frame < 1000; frame++) {
    stream << update_buffer(buffer)
           << process(buffer)
           << synchronize();
}

// Avoid: Reallocating each frame
for (int frame = 0; frame < 1000; frame++) {
    Buffer<float> buffer = device.create_buffer<float>(size);
    // ...
}
```

### 3. Use Appropriate Stream Types

```cpp
// Compute work on compute stream
Stream compute = device.create_stream(StreamTag::COMPUTE);

// Transfer work on copy stream (may overlap with compute)
Stream transfer = device.create_stream(StreamTag::COPY);

transfer << upload_data(buffer);
compute << process(buffer);
// Upload and compute can overlap!
```

### 4. Minimize Host-GPU Transfers

```cpp
// Good: Keep data on GPU
Buffer<float> persistent = device.create_buffer<float>(n);
for (int i = 0; i < 100; i++) {
    stream << kernel(persistent).dispatch(n);
}
stream << persistent.copy_to(host);

// Avoid: Round-trips
for (int i = 0; i < 100; i++) {
    stream << buffer.copy_from(host_data)  // Upload
           << kernel(buffer).dispatch(n)
           << buffer.copy_to(host_data)    // Download
           << synchronize();
}
```
