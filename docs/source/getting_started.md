# Getting Started

This guide will help you get up and running with LuisaCompute, from installation to your first GPU-accelerated program.

## Overview

LuisaCompute is a high-performance cross-platform computing framework designed for modern graphics and general-purpose GPU computing. It balances three key goals:

- **Unification**: Write code once, run on CUDA, DirectX, Metal, Vulkan, HIP, or the fallback backend
- **Programmability**: Modern C++ embedded DSL with intuitive syntax
- **Performance**: Optimized backends with automatic command scheduling

The framework consists of three major components:
1. **Embedded DSL**: Write GPU kernels directly in C++ using `Var<T>`, `Kernel`, and `Callable`
2. **Unified Runtime**: Resource management and command scheduling with `Context`, `Device`, and `Stream`
3. **Multiple Backends**: CUDA, DirectX, Metal, Vulkan, HIP, and fallback implementations

## Prerequisites

### Hardware Requirements

| Backend | Requirements |
|---------|-------------|
| CUDA | NVIDIA GPU with RTX support (RTX 20 series or newer recommended) |
| DirectX | DirectX 12.1 & Shader Model 6.5 compatible GPU |
| Metal | macOS device with Metal support |
| Fallback | A platform supported by the configured LLVM and Embree packages |

### Software Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS 12+
- **Compiler**: C++20 compatible compiler (MSVC 2019+, GCC 11+, Clang 14+)
- **Build System**: CMake 3.20+ or XMake 2.7+
- **Python**: 3.10+ (for bootstrap script and Python frontend)

#### Backend-Specific Requirements

- **CUDA**: CUDA Toolkit 11.7+, NVIDIA driver R535+
- **DirectX**: Windows SDK 10.0.19041.0+
- **Metal**: Xcode 14+ (macOS)
- **Fallback**: LLVM and Embree development packages

## Building from Source

### Clone the Repository

```bash
git clone -b next https://github.com/LuisaGroup/LuisaCompute.git --recursive
cd LuisaCompute
```

> **Important**: The `--recursive` flag is required to fetch all submodules.

### Build Using the Bootstrap Script (Recommended)

The bootstrap script automates dependency installation and building:

```bash
# Build with CUDA backend using CMake
python bootstrap.py cmake -f cuda -b

# Build with all available backends
python bootstrap.py cmake -f all -b

# Build with specific configuration
python bootstrap.py cmake -f cuda -b -- -DCMAKE_BUILD_TYPE=Release

# Generate IDE project without building
python bootstrap.py cmake -f cuda -c -o cmake-build-release
```

Install missing dependencies:
```bash
python bootstrap.py -i cmake     # Install/upgrade CMake
python bootstrap.py -i xmake     # Install XMake
```

### Build Using CMake Directly

```bash
mkdir build && cd build
cmake .. -DLUISA_BACKEND_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### Build Using XMake

```bash
xmake config --cuda=true
xmake build
```

### CMake Integration in Your Project

```cmake
# Add LuisaCompute as a subdirectory
add_subdirectory(LuisaCompute)

# Link to your target
target_link_libraries(your_target PRIVATE luisa::compute)
```

Or use the [starter template](https://github.com/LuisaGroup/CMakeStarterTemplate) for quick setup.

## Your First Program

Let's create a simple program that fills an image with a gradient color on the GPU.

### Complete Example

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>  // For $if, $while, etc.
#include <stb_image_write.h>   // For saving the result

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    // Step 1: Initialize the runtime context
    Context context{argv[0]};
    
    // Step 2: Create a device (GPU) to run computations on
    Device device = context.create_device("cuda");
    
    // Step 3: Create a command stream for submitting work
    Stream stream = device.create_stream();
    
    // Step 4: Create a 1024x1024 image with 4-channel 8-bit storage
    // The template argument <float> means pixel values are automatically
    // converted between byte4 and float4 when reading/writing
    Image<float> image = device.create_image<float>(
        PixelStorage::BYTE4,  // Internal storage format
        1024, 1024            // Width and height
    );
    
    // Step 5: Define a callable (reusable function)
    Callable linear_to_srgb = [](Float4 linear) noexcept {
        auto x = linear.xyz();
        // Apply gamma correction using component-wise select
        return make_float4(
            select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                   12.92f * x,
                   x <= 0.00031308f),
            linear.w);
    };
    
    // Step 6: Define a 2D kernel (entry point for GPU execution)
    Kernel2D fill_kernel = [&](ImageFloat image) noexcept {
        // Get the current thread's position in the grid
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        
        // Calculate normalized UV coordinates (0 to 1)
        Float2 uv = make_float2(coord) / make_float2(size);
        
        // Create a gradient color
        Float4 color = make_float4(uv, 0.5f, 1.0f);
        
        // Apply sRGB conversion and write to image
        image->write(coord, linear_to_srgb(color));
    };
    
    // Step 7: Compile the kernel into a shader
    auto shader = device.compile(fill_kernel);
    
    // Step 8: Prepare host memory for downloading the result
    std::vector<std::byte> pixels(1024 * 1024 * 4);
    
    // Step 9: Execute commands on the GPU
    stream << shader(image.view(0)).dispatch(1024, 1024)  // Launch kernel
           << image.copy_to(pixels.data())                 // Download to CPU
           << synchronize();                               // Wait for completion
    
    // Step 10: Save the result
    stbi_write_png("output.png", 1024, 1024, 4, pixels.data(), 0);
    
    return 0;
}
```

### Building Your Program

```bash
# Compile with your build system, linking against LuisaCompute
g++ -std=c++20 myprogram.cpp -I/path/to/luisa/include \
    -L/path/to/luisa/lib -lluisa-compute-runtime \
    -o myprogram
```

## Core Concepts

### 1. Context and Device

The **Context** is the entry point for the runtime. It manages backend plugins and global configuration:

```cpp
// Create context with executable path
Context context{argv[0]};

// Create a specific backend device
Device cuda_device = context.create_device("cuda");
Device dx_device = context.create_device("dx");     // Windows only
Device metal_device = context.create_device("metal"); // macOS only
Device fallback_device = context.create_device("fallback");

// Or use the default available backend
Device device = context.create_default_device();
```

### 2. Streams and Commands

**Streams** are command queues for asynchronous execution:

```cpp
// Create a compute stream
Stream stream = device.create_stream(StreamTag::COMPUTE);

// Commands are submitted using operator<<
stream << command1 << command2 << command3;

// Commands are automatically batched and executed
// Explicit synchronization:
stream << synchronize();
// Or:
stream.synchronize();
```

### 3. Resources

#### Buffers

Linear memory for structured data:

```cpp
// Create a buffer of 1000 float4 elements
Buffer<float4> buffer = device.create_buffer<float4>(1000);

// Upload data from host
std::vector<float4> host_data(1000);
stream << buffer.copy_from(host_data.data());

// Download data to host
stream << buffer.copy_to(host_data.data());
```

#### Images

2D textures with hardware-accelerated sampling:

```cpp
// Create a 1920x1080 image with RGBA8 format
Image<float> image = device.create_image<float>(
    PixelStorage::BYTE4, 1920, 1080);

// Create with mipmaps
Image<float> image_with_mips = device.create_image<float>(
    PixelStorage::BYTE4, 1920, 1080, 10);  // 10 mipmap levels
```

### 4. The DSL

#### Variables

Use `Var<T>` to represent device-side variables:

```cpp
// Scalar types
Var<float> x = 1.0f;           // Or: Float x = 1.0f;
Var<int> count = 100;          // Or: Int count = 100;

// Vector types
Float2 pos = make_float2(1.0f, 2.0f);
Float3 color = make_float3(1.0f, 0.5f, 0.0f);
Float4 rgba = make_float4(color, 1.0f);

// Swizzling
Float2 xy = pos.xy();          // Extract x and y
Float3 xyz = rgba.xyz();       // Extract first three components
Float4 repeated = pos.xxxx();  // (x, x, x, x)
```

#### Control Flow

Use `$`-prefixed macros for device-side control flow:

```cpp
$if (condition) {
    // GPU-side branch
} $elif (other_condition) {
    // Another branch
} $else {
    // Default branch
};

$while (condition) {
    // Loop on GPU
};

$for (i, 0, 100) {
    // Loop from 0 to 99
    // i is Var<int>
};

$for (i, 0, 100, 2) {
    // Loop from 0 to 98 with step 2
};
```

> **Note**: Use native C++ `if` for host-side (compile-time) decisions that affect what gets recorded into the kernel.

#### Kernels and Callables

```cpp
// A Callable is a reusable device function
Callable lerp = [](Float a, Float b, Float t) noexcept {
    return a * (1.0f - t) + b * t;
};

// A Kernel is the entry point for GPU execution
Kernel2D render_kernel = [&](ImageFloat image, BufferFloat4 colors) noexcept {
    Var coord = dispatch_id().xy();
    Var index = coord.y * dispatch_size().x + coord.x;
    
    Float4 color = colors.read(index);
    image->write(coord, color);
};

// Compile and dispatch
auto shader = device.compile(render_kernel);
stream << shader(image, colors).dispatch(1024, 1024);
```

## Python Frontend

LuisaCompute also provides a Python frontend that exposes the same GPU computing capabilities with Pythonic syntax. The Python frontend uses the `@func` decorator (analogous to C++ `Kernel`/`Callable`) to define device-side functions and traces them into the same IR as the C++ DSL.

### Installation

Install the pre-built package from PyPI (Python 3.10+ required):

```bash
python -m pip install luisa-python
```

Or build from source:

```bash
python -m pip wheel <path-to-LuisaCompute> -w <output-dir>
```

### Hello World in Python

The following program fills an image with a gradient color — the same task as the C++ example above:

```python
from luisa import *
from luisa.types import *

# Initialize the runtime (auto-selects the best available backend)
init()

# Create a 1024x1024 image with 4-channel 8-bit storage
res = 1024, 1024
img = Image2D(*res, 4, float, storage="BYTE")

# Define a kernel using the @func decorator
@func
def fill():
    coord = dispatch_id().xy
    size = dispatch_size().xy
    uv = (float2(coord) + 0.5) / float2(size)
    img.write(coord, float4(uv, 0.5, 1.0))

# Dispatch the kernel and save the result
fill(dispatch_size=(*res, 1))
img.to_image("gradient.png")
```

To specify a particular backend, pass its name to `init()`:

```python
init(backend_name="cuda")   # NVIDIA GPU
init(backend_name="dx")     # DirectX (Windows)
init(backend_name="metal")  # Metal (macOS)
init(backend_name="fallback") # Native C++ LLVM/Embree fallback
```

You can also set the `LUISA_BACKEND` environment variable to select the backend without modifying code.

### Key Differences from C++

The Python frontend mirrors the C++ DSL but follows Python idioms:

| Concept | C++ | Python |
|---------|-----|--------|
| Kernel definition | `Kernel2D k = [&](...) { ... };` | `@func` decorator |
| Variable declaration | `Float x = 1.0f;` | `x = 1.0` (traced automatically) |
| Vector construction | `make_float2(1.0f, 2.0f)` | `float2(1.0, 2.0)` |
| Swizzling | `v.xy()` | `v.xy` (property, no parentheses) |
| Control flow | `$if`, `$while`, `$for` | `if`, `while`, `for` (Python native) |
| Device init | `Context ctx{argv[0]}; Device d = ctx.create_device("cuda");` | `init()` |
| Dispatch | `stream << shader(args).dispatch(w, h);` | `kernel(dispatch_size=(w, h, 1))` |
| Synchronization | `stream << synchronize();` | `synchronize()` |
| Resource capture | Captured by reference in lambda | Captured by closure automatically |

> **Note**: In the Python frontend, structures and arrays are passed by **reference** to `@func`, while scalar, vector, and matrix types are passed by **value**. This follows Python's convention where mutable objects are reference types.

### Buffers and Data Transfer

Use `Buffer` for linear device memory. Data transfers use NumPy arrays:

```python
import numpy as np
from luisa import *
from luisa.types import *

init()

# Create a buffer of 1024 float values
buf = Buffer(1024, float)

# Upload data from a NumPy array
host_data = np.arange(1024, dtype=np.float32)
buf.copy_from(host_data)

# Run a kernel that modifies the buffer
@func
def double_values():
    i = dispatch_id().x
    val = buf.read(i)
    buf.write(i, val * 2.0)

double_values(dispatch_size=1024)

# Download results back to the host
result = np.zeros(1024, dtype=np.float32)
buf.copy_to(result)
synchronize()
```

### Custom Structures

Define custom data structures with `StructType`:

```python
from luisa import *
from luisa.types import *

init()

# Define a struct type with named fields
Particle = StructType(position=float3, velocity=float3, mass=float)

# Create a buffer of structs
particles = Buffer(256, Particle)

@func
def update_particles(dt: float):
    i = dispatch_id().x
    p = particles.read(i)
    p.position = p.position + p.velocity * dt
    particles.write(i, p)

update_particles(0.016, dispatch_size=256)
```

### Images and the GUI Window

The Python frontend provides `Image2D` for texture operations and a built-in `GUI` for real-time display:

```python
from luisa import *
from luisa.types import *

init()

res = 512, 512
display = Image2D(*res, 4, float, storage="BYTE")

@func
def render(time: float):
    coord = dispatch_id().xy
    size = dispatch_size().xy
    uv = (float2(coord) + 0.5) / float2(size)
    # Animated color pattern
    r = sin(uv.x * 6.28 + time) * 0.5 + 0.5
    g = sin(uv.y * 6.28 + time * 1.3) * 0.5 + 0.5
    display.write(coord, float4(r, g, 0.5, 1.0))

gui = GUI("My Window", res)
t = 0.0
while gui.running():
    render(t, dispatch_size=(*res, 1))
    gui.set_image(display)
    dt = gui.show()
    t += dt / 1000.0
synchronize()
```

### Automatic Differentiation

The Python frontend supports reverse-mode automatic differentiation using the `autodiff` context manager:

```python
from luisa import *
from luisa.autodiff import *
from luisa.types import *
import numpy as np

init()

N = 1024
x_buf = Buffer(N, float)
dx_buf = Buffer(N, float)

x_values = np.arange(N, dtype=np.float32)
x_buf.copy_from(x_values)

@func
def compute_grad():
    i = dispatch_id().x
    x = x_buf.read(i)
    with autodiff():
        requires_grad(x)
        y = x * x + sin(x)  # f(x) = x² + sin(x)
        backward(y)          # compute dy/dx
        dx = grad(x)         # retrieve gradient
    dx_buf.write(i, dx)

compute_grad(dispatch_size=N)

result = np.zeros(N, dtype=np.float32)
dx_buf.copy_to(result)
synchronize()
# result[i] ≈ 2*x[i] + cos(x[i])
```

### Ray Tracing

The Python frontend supports hardware-accelerated ray tracing via `Accel`:

```python
from luisa import *
from luisa.builtin import *
from luisa.types import *
import numpy as np

init()

res = 512, 512
image = Image2D(*res, 4, float, storage="BYTE")

# Create geometry
vertex_buffer = Buffer(3, float3)
index_buffer = Buffer(3, int)
vertex_buffer.copy_from([
    float3(-0.5, -0.5, -2.0),
    float3( 0.5, -0.5, -1.5),
    float3( 0.0,  0.5, -1.0),
])
index_buffer.copy_from(np.array([0, 1, 2], dtype=np.int32))

# Build acceleration structure
accel = Accel()
accel.add(vertex_buffer, index_buffer)
accel.update()

@func
def raytrace(image, accel):
    coord = dispatch_id().xy
    p = (float2(coord) + 0.5) / float2(dispatch_size().xy) * 2.0 - 1.0
    ray = make_ray(
        float3(p * float2(1.0, -1.0), 0.0),  # origin
        float3(0.0, 0.0, -1.0),               # direction
        0.0, 1000.0)                           # t_min, t_max
    hit = accel.trace_closest(ray, -1)
    color = float3(0.3, 0.5, 0.7)  # sky
    if hit.hitted():
        color = hit.interpolate(
            float3(1, 0, 0),  # vertex 0 color
            float3(0, 1, 0),  # vertex 1 color
            float3(0, 0, 1))  # vertex 2 color
    image.write(coord, float4(color, 1.0))

raytrace(image, accel, dispatch_size=(*res, 1))
image.to_image("raytrace.png")
```

### Shared Memory and Block Synchronization

The Python frontend supports shared (threadgroup) memory via `SharedArrayType` and block synchronization via `sync_block()`:

```python
from luisa import *
from luisa.builtin import *
from luisa.types import *

init()

block_size = 256
SharedArray = SharedArrayType(float, block_size)

buf = Buffer(1024, float)

@func
def reduce_kernel():
    set_block_size(block_size, 1, 1)
    shared = SharedArray()
    tid = thread_id().x
    gid = dispatch_id().x
    shared[tid] = buf.read(gid)
    sync_block()
    # Parallel reduction within the block
    s = block_size // 2
    while s > 0:
        if tid < s:
            shared[tid] = shared[tid] + shared[tid + s]
        sync_block()
        s = s // 2
    if tid == 0:
        buf.write(block_id().x, shared[0])

reduce_kernel(dispatch_size=1024)
```

### Python Example Programs

Many more examples are available in `src/tests/python/`:

| File | Description |
|------|-------------|
| `test-helloworld.py` | Gradient image (minimal example) |
| `test-buffer.py` | Buffer read/write with custom structs |
| `test-texture.py` | Image loading, flipping, and GUI display |
| `test-rtx.py` | Real-time ray tracing with animated geometry |
| `test-game-of-life.py` | Conway's Game of Life simulation |
| `test-autodiff.py` | Reverse-mode automatic differentiation |
| `test-path-tracing.py` | Full path tracer with multiple bounces |
| `test-sdf-renderer.py` | Signed distance field renderer |
| `test-shadertoy.py` | ShaderToy-style procedural rendering |
| `test-mpm.py` | Material Point Method simulation |
| `test-indirect.py` | GPU-driven indirect dispatch |
| `test-pytorch-interop.py` | Interoperability with PyTorch tensors |
| `test-bindless.py` | Bindless resource arrays |

## Next Steps

- Learn more about the [DSL](dsl.md) for advanced kernel programming
- Explore [Resources and Runtime](resources.md) for detailed resource management
- Understand the [Architecture](architecture.md) for implementation details
- Check out the [API Reference](api_reference.rst) for complete class documentation
- Browse example programs in `src/tests` for real-world usage patterns

## Troubleshooting

### Backend Not Found

```
Error: Backend 'cuda' not found
```

- Ensure the backend plugin exists: `luisa-backend-cuda.dll` (Windows) or `luisa-backend-cuda.so` (Linux)
- Check that the plugin is in the same directory as your executable
- Verify CUDA toolkit is installed and drivers are up to date

### Compilation Errors

- Ensure C++20 is enabled (`-std=c++20` for GCC/Clang, `/std:c++20` for MSVC)
- Check that all submodules are cloned (`git submodule update --init --recursive`)

### Runtime Validation

Enable validation layer for debugging:
```cpp
Device device = context.create_device("cuda", nullptr, true);
// Or via environment variable:
// LUISA_ENABLE_VALIDATION=1 ./myprogram
```
