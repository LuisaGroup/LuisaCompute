# LuisaCompute

![teaser](https://user-images.githubusercontent.com/7614925/195987646-fe932ebe-ca6e-4d6e-ab2a-203bcfd3d559.jpg)

LuisaCompute is a high-performance cross-platform computing framework for graphics and beyond.

LuisaCompute is also the *rendering framework* described in the **SIGGRAPH Asia 2022** paper
> ***LuisaRender: A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures***.

See also [LuisaRender](https://github.com/LuisaGroup/LuisaRender) for the *rendering application* as described in the paper; and please visit the [project page](https://luisa-render.com) for other information about the paper and the project.

## Table of Contents

* [Overview](#overview)
    + [Embedded Domain-Specific Language](#embedded-domain-specific-language)
    + [Unified Runtime with Resource Wrappers](#unified-runtime-with-resource-wrappers)
    + [Multiple Backends](#multiple-backends)
    + [Python Frontend](#python-frontend)
    + [C API and Frontends in Other Languages](#c-api-and-frontends-in-other-languages)
* [Building](#building)
* [Usage](#usage)
    + [A Minimal Example](#a-minimal-example)
    + [Basic Types](#basic-types)
    + [Structures](#structures)
    + [Built-in Functions](#built-in-functions)
    + [Control Flows](#control-flows)
    + [Callable and Kernels](#callable-and-kernels)
    + [Backends, Context, Devices, and Resources](#devices-and-resources)
    + [Command Submission and Synchronization](#command-submission-and-synchronization)
* [Applications](#applications)
* [Documentation and Tutorials](#documentation-and-tutorials)
* [Roadmap](#roadmap)
* [Citation](#citation)


## Overview

LuisaCompute seeks to balance the seemingly ever-conflicting pursuits for ***unification***, ***programmability***, and ***performance***. To achieve this goal, we design three major components:
- A domain-specific language (DSL) embedded inside modern C++ for kernel programming exploiting JIT code generation and compilation;
- A unified runtime with resource wrappers for cross-platform resource management and command scheduling; and
- Multiple optimized backends, including CUDA, DirectX, Metal, LLVM, and ISPC.

To demonstrate the practicality of the system, we also build a Monte Carlo renderer, [LuisaRender](https://github.com/LuisaGroup/LuisaRender), atop the framework, which is faster than the state-of-the-art rendering frameworks on modern GPUs.

### Embedded Domain-Specific Language

The DSL in our system provides a unified approach to authoring kernels, i.e., programmable computation tasks on the device. Distinct from typical graphics APIs that use standalone shading languages for device code, our system unifies the authoring of both the host-side logic and device-side kernels into the same language, i.e., modern C++.

The implementation purely relies on the C++ language itself, without any custom preprocessing pass or compiler extension. We exploit meta-programming techniques to simulate the syntax, and function/operator overloading to dynamically trace the user-defined kernels. ASTs are constructed during the tracing as an intermediate representation and later handed over to the backends for generating concrete, platform-dependent shader source code.

Example program in the embedded DSL:
```cpp
Callable to_srgb = [](Float3 x) {
    $if (x <= 0.00031308f) {
        x = 12.92f * x;
    } $else {
        x = 1.055f * pow(x, 1.f / 2.4f) - .055f;
    };
    return x;
};
Kernel2D fill = [&](ImageFloat image) {
    auto coord = dispatch_id().xy();
    auto size = make_float2(dispatch_size().xy());
    auto rg = make_float2(coord) / size;
    // invoke the callable
    auto srgb = to_srgb(make_float3(rg, 1.f));
    image.write(coord, make_float4(srgb, 1.f));
};
```

### Unified Runtime with Resource Wrappers

Like the RHIs in game engines, we introduce an abstract runtime layer to re-unify the fragmented graphics APIs across platforms. It extracts the common concepts and constructs shared by the backend APIs and plays the bridging role between the high-level frontend interfaces and the low-level backend implementations.

On the programming interfaces for users, we provide high-level resource wrappers to ease programming and eliminate boilerplate code. They are strongly and statically typed modern C++ objects, which not only simplify the generation of commands via convenient member methods but also support close interaction with the DSL. Moreover, with the resource usage information in kernels and commands, the runtime automatically probes the dependencies between commands and re-schedules them to improve hardware utilization.

### Multiple Backends

The backends are the final realizers of computation. They generate concrete shader sources from the ASTs and compile them into native shaders. They implement the virtual device interfaces with low-level platform-dependent API calls and translate the intermediate command representations into native kernel launches and command dispatches.

Currently, we have 5 backends, including 3 GPU backends based on CUDA, Metal, and DirectX, respectively, a scalar CPU backend on LLVM, and a vectorized CPU backend on ISPC.

### Python Frontend

Besides the native C++ DSL and runtime interfaces, we are also working on a Python frontend. See [README_Python.md](README_Python.md) (in Chinese; we are translating it to English) for more information. Examples using the Python frontend can be found under `src/py`.

Example program with the Python frontend:
```python
import luisa
from luisa.mathtypes import *
from cv2 import imwrite

n = 2048
luisa.init()
image = luisa.Texture2D.zeros((2 * n, n), 4, float)

@luisa.func # makes LuisaRender handle the function
def draw(max_iter):
    p = dispatch_id().xy
    z, c = float2(0), 2 * p / n - float2(2, 1)
    for itr in range(max_iter):
        z = float2(z.x**2 - z.y**2, 2 * z.x * z.y) + c
        if length(z) > 20:
            break
    image.write(p, float4(float3(1 - itr/max_iter), 1))

draw(50, dispatch_size=(2*n, n)) # parallelize
imwrite("mandelbrot.exr", image.numpy())
```

> ⚠️ The Python frontend is mostly functional but still under active development, possibly with regressions and breaking changes from time to time.

### C API and Frontends in Other Languages

We are also making a C API for creating other language bindings and frontends (e.g., in [Rust](https://github.com/LuisaGroup/luisa-compute-rs) and C#).

## Building

> Note: LuisaCompute is a *rendering framework* rather than a *renderer* itself. It is designed to provide general computation functionalities on modern stream-processing hardware, on which high-performance, cross-platform graphics applications can be easily built. If you would like to just try out a Monte Carlo renderer out of the box rather than building one from the scratch, please see [LuisaRender](https://github.com/LuisaGroup/LuisaRender).

LuisaCompute follows the standard CMake build process. Basically these steps:

- Check your hardware and platform. Currently, we support CUDA on Linux and Windows; DirectX on Windows; Metal on macOS; and ISPC and LLVM on all the major platforms. For CUDA and DirectX, an RTX-enabled graphics card, e.g., NVIDIA RTX 20 and 30 series, is required.

- Prepare the environment and dependencies. We recommend using the latest IDEs, Compilers, CMake, CUDA drivers, etc. Since we aggressively use new technologies like C++20 and OptiX 7.1+, you may need to, for example, upgrade your VS to 2019 or 2022, and install CUDA 11.0+. For some tests like the toy path tracer, [OpenCV](opencv.org) is also required.

- Clone the repo with the `--recursive` option:
    ```bash
    git clone --recursive https://github.com/LuisaGroup/LuisaCompute.git
    ```
  Since we use Git submodules to manage third-party dependencies, a `--recursive` clone is required.

- Configure the project using CMake. E.g., for command line, `cd` into the project folder and type `cmake -S . -B <build-folder>`. You might also want to specify your favorite generators and build types using options like `-G Ninja` and `-D CMAKE_BUILD_TYPE=Release`. A typical, full command sequence for this would be like
    ```bash
    cd LuisaCompute
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
    ```

- If the configuration succeeds, you are now able to build the project. Type `cmake --build <build-folder>` in the command line, or push the build button if you generated, e.g., a VS project. (And in case the configuration step unluckily failed :-(, please file an [issue](https://github.com/LuisaGroup/LuisaCompute/issues)).

- After building, you will find the binaries under `<build-folder>/bin`. You can now play around with them, for example type `<build-folder>/bin/test_texture_io cuda` to generate a gradient color texture with the CUDA backend.

See also [BUILD.md](BUILD.md) for details on platform requirements, configuration options, and other precautions.

## Usage

### A Minimal Example

Using LuisaCompute to construct a graphics application basically involves the following steps:

1. Create a `Context` and loading a `Device` plug-in;
2. Create a `Stream` for command submission and other device resources (e.g., `Buffer<T>`s for linear storage, `Image<T>`s for 2D readable/writable textures, and `Mesh`es and `Accel`s for ray-scene intersection testing structures) via `Device`'s `create_*` interfaces;
3. Author `Kernel`s to describe the on-device computation tasks, and compile them into `Shader`s via `Device`'s `compile` interface;
4. Generate `Command`s via each resource's interface (e.g., `Buffer<T>::copy_to`), or `Shader`'s `operator()` and `dispatch`, and submit them to the stream;
5. Wait for the results by inserting a `synchronize` phoney command to the `Stream`.

Putting the above together, a minimal example program that write gradient color to an image would look like
```cpp

#include <luisa-compute.h>
#include <dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    // Step 1.1: Create a context
    Context context{argv[0]};
    
    // Step 1.2: Load the CUDA backend plug-in and create a device
    auto device = context.create_device("cuda");
    
    // Step 2.1: Create a stream for command submission
    auto stream = device.create_stream();
    
    // Step 2.2: Create an 1024x1024 image with 4-channel 8-bit storage for each pixel; the template 
    //           argument `float` indicates that pixel values reading from or writing to the image
    //           are converted from `byte4` to `float4` or `float4` to `byte4` automatically
    auto device_image = device.create_image<float>(PixelStorage::BYTE4, 1024u, 1024u, 0u);
    
    // Step 3.1: Define kernels to describe the device-side computation
    // 
    //           A `Callable` is a function *entity* (not directly inlined during 
    //           the AST recording) that is invocable from kernels or other callables
    Callable linear_to_srgb = [](Float4 /* alias for Var<float4> */ linear) noexcept {
        // The DSL syntax is much like the original C++
        auto x = linear.xyz();
        return make_float4(
            select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                   12.92f * x,
                   x <= 0.00031308f),
            linear.w);
    };
    //           A `Kernel` is an *entry* function to the device workload 
    Kernel2D fill_image_kernel = [&linear_to_srgb](ImageFloat /* alias for Var<Image<float>> */ image) noexcept {
        Var coord = dispatch_id().xy();
        Var rg = make_float2(coord) / make_float2(dispatch_size().xy());
        image.write(coord, linear_to_srgb(make_float4(rg, 1.0f, 1.0f)));
    };
    
    // Step 3.2: Compile the kernel into a shader (i.e., a runnable object on the device)
    auto fill_image = device.compile(fill_image_kernel);
    
    // Prepare the host memory for holding the image
    std::vector<std::byte> download_image(1024u * 1024u * 4u);
    
    // Step 4: Generate commands from resources and shaders, and
    //         submit them to the stream to execute on the device
    stream << fill_image(device_image.view(0)).dispatch(1024u, 1024u)
           << device_image.copy_to(download_image.data())
           << synchronize();// Step 5: Synchronize the stream
   
   // Now, you have the device-computed pixels in the host memory!
   your_image_save_function("color.png", downloaded_image, 1024u, 1024u, 4u);
}
```

### Basic Types

In addition to standard C++ scalar types (e.g., `int`, `uint` --- alias of `uint32_t`, `float`, and `bool`), LuisaCompute provides vector/matrix types for 3D graphics, including the following types:
```cpp
// boolean vectors
using bool2 = Vector<bool, 2>;   // alignment: 2B
using bool3 = Vector<bool, 3>;   // alignment: 4B
using bool4 = Vector<bool, 4>;   // alignment: 4B
// signed and unsigned integer vectors
using int2 = Vector<int, 2>;     // alignment: 8B
using int3 = Vector<int, 3>;     // alignment: 16B
using int4 = Vector<int, 4>;     // alignment: 16B
using uint2 = Vector<uint, 2>;   // alignment: 8B
using uint3 = Vector<uint, 3>;   // alignment: 16B
using uint4 = Vector<uint, 4>;   // alignment: 16B
// floating-point vectors and matrices
using float2 = Vector<float, 2>; // alignment: 8B
using float3 = Vector<float, 3>; // alignment: 16B
using float4 = Vector<float, 4>; // alignment: 16B
using float2x2 = Matrix<2>;      // column-major, alignment: 8B
using float3x3 = Matrix<3>;      // column-major, alignment: 16B
using float4x4 = Matrix<4>;      // column-major, alignment: 16B
```

> ⚠️ Please pay attention to the alignment of 3D vectors and matrices --- they are aligned like 4D ones rather than packed. Also, we do not provide 64-bit integer or floating-point vector/matrix types, as they are less useful and typically unsupported on GPUs.

To make vectors/matrices, we provide `make_*` and read-only swizzle interfaces, e.g.,
```cpp
auto a = make_float2();       // (0.f, 0.f)
auto b = make_int3(1);        // (1,   1,   1)
auto c = make_uint3(b);       // (1u,  1u,  1u): converts from a same-dimentional but (possibly) differently typed vector
auto d = make_float3(a, 1.f); // (0.f, 0.f, 1.f): construct float3 from float2 and a float scalar
auto e = d.zzxy();            // (1.f, 1.f, 0.f, 0.f): swizzle
auto m = make_float2x2(1.f);  // ((1.f, 0.f,), (0.f, 1.f)): diagonal matrix from a scalar
...
```

Operators are also overloaded for scalar-vector, vector-vector, scalar-matrix, vector-matrix, and matrix-matrix calculations, e.g.,
```cpp
auto one = make_float2(1.f); // (1.f, 1.f)
auto two = 2.f;
auto three = one + two;      // (3.f, 3.f), scalar broadcast to vector
auto m2 = make_float2(2.f);  // ((2.f, 0.f), (0.f, 2.f))
auto m3 = 1.5f * m2;         // ((3.f, 0.f), (0.f, 3.f)), scalar-matrix multiplication
auto v = m3 * one;           // (3.f, 3.f), matrix-vector multiplication, the vector should always
                             // appear at the right-hand side and is interpreted as a column vector
auto m6 = m2 * m3;           // ((6.f, 0.f), (0.f, 6.f)), matrix-matrix multiplication
```

The scalar, vector, matrix, and array types are also supported in the DSL, together with `make_*`, swizzles, and operators. Just wrap them in the `Var<T>` template or use the pre-defined aliases:
```cpp
// scalar types; note that 64-bit ones are not supported
using Int = Var<int>;
using UInt = Var<uint>;
using Float = Var<float>;
using Bool = Var<bool>;

// vector types
using Int2 = Var<int2>; // = Var<Vector<int, 2>>
using Int3 = Var<int3>; // = Var<Vector<int, 3>>
/* ... */

// matrix types
using Float2x2 = Var<float2x2>; // = Var<Matrix<2>>
using Float3x3 = Var<float3x3>; // = Var<Matrix<3>>
using Float4x4 = Var<float4x4>; // = Var<Matrix<4>>

// array types
template<typename T, size_t N>
using ArrayVar = Var<std::array<T, N>>;

// make_*
auto a = make_float2(one);    // Float2(1.f, 1.f), suppose one = Float(1.f)
auto m = make_float2x2(a, a); // Float2x2((1.f, 1.f), (1.f, 1.f))
auto c = make_int2(a);        // Int2(1, 1)
auto d = c.xxx();             // Int3(1, 1, 1)
auto e = d[0];                // 1
/* ... */

// operators
auto v2 = a * 2.f;  // Float2(2.f, 2.f)
auto eq = v2 == v2; // Bool2(true, true)
/* ... */
```

> ⚠️ The only exception is that we disable `operator&&` and `operator||` in the DSL. This is because the DSL does not support the *short-circuit* semantics. We disable them to avoid ambiguity. Please use `operator&` and `operator|` instead, which have the consistent non-short-circuit semantics on both the host and device sides.

Besides the `Var<T>` template, there's also an `Expr<T>`, which is to `Var<T>` what `const T &` is to `T` on the host side. In other words, `Expr<T>` stands for a const DSL variable reference, which does not create variables copies when passed around. However, note that the parameters of `Callable`/`Kernel` definition functions may only be `Var<T>`. This restriction might be removed in the future.

To conveniently convert a C++ variable to the DSL, we provide a helper template function `def<T>`:
```cpp
auto a = def(1.f);              // equivalent to auto a = def<float>(1.f);
auto b_host = make_float2(1.f); // host C++ variable float2(1.f, 1.f)
auto b_device = def(b_host);    // device DSL variable Float2(1.f, 1.f)
/* ... */
```

### Structures

To export a C++ data struct to the DSL, we provide a helper macro `LUISA_STRUCT`, which (semi-)automatically reflects the member layouts of the input structure:
```cpp
// A C++ data structure
namespace foo {
struct alignas(8) S {
    float a;
    int   b;
};
}

// A reflected DSL structure
LUISA_STRUCT(foo::S, a, b) {
/* device-side member functions, e.g., */
    [[nodiscard]] auto twice_a() const noexcept { return 2.f * a; }
};
```

> ⚠️ The `LUISA_STRUCT` may only be used in the global namespace. The C++ structure to be exported may only contain scalar, vector, matrix, array, and other already exported structure types. The alignment of the *whole* structure specified with `alignas` will be reflected but must be under 16B; member alignments specified with `alignas` are not supported.

### Built-in Functions

For the DSL, we provide a rich set of built-in functions, in the following categories
- Thread coordinate and launch configuration queries, including `block_id`, `thread_id`, `dispatch_size`, and `dispatch_id`;
- Mathematical routines, such as `max`, `abs`, `sin`, `pow`, and `sqrt`;
- Resource accessing and modification methods, such as texture sampling, buffer read/write, and ray intersection;
- Variable construction and type conversion, e.g., the aforementioned `make_*`, `cast<T>` for static type casting, and `as<T>` for bitwise type casting; and
- Optimization hints for backend compilers, which currently consist of `assume` and `unreachable`.

The mathematical functions basically mirrors [GLSL](https://www.khronos.org/opengl/wiki/Core_Language_(GLSL)). We are working on the documentations that will provide more descriptions on them.

### Control Flows

The DSL in LuisaCompute supports device-side control flows. They are provided as special macros prefixed with `$`:
```cpp
$if (cond) { /*...*/ };
$if (cond) { /*...*/ } $else { /*...*/ };
$if (cond) { /*...*/ } $elif (cond2) { /*...*/ };
$if (cond) { /*...*/ } $elif (cond2) { /*...*/ } $else { /*...*/ };

$while (cond) { /*...*/ };
$for (variable, n) { /*...*/ };
$for (variable, begin, end) { /*...*/ };
$for (variable, begin, end, step) { /*...*/ };
$loop { /*...*/ }; // infinite loop, unless $break'ed

$switch (variable) {
    $case (value) { /*...*/ }; // no $break needed inside, as we automatically add one
    $default { /*...*/ };      // no $break needed inside, as we automatically add one
};

$break;
$continue;
```

Note that users are still able to use the *native* C++ control flows, i.e., `if`, `while`, etc. *without* the `$` prefix. In that case the *native* control flows acts like a *meta-stage* to the DSL that directly controls the generation of the callables/kernels. This can be a powerful means to achieve *multi-stage programming* patterns. Such usages can be found throughout [LuisaRender](https://github.com/LuisaGroup/LuisaRender). We will cover such usage in the tutorials in the future.

### Callable and Kernels

LuisaCompute supports two categories of device functions: `Kernel`s (`Kernel1D`, `Kernel2D`, or `Kernel3D`) and `Callable`s. Kernels are entries to the parallelized computation tasks on the device (equivelant to CUDA's `__global__` functions). Callables are function objects invocable from kernels or other callables (i.e., like CUDA's `__device__` functions). Both kinds are template classes that are constructible from C++ functions or function objects including lambda expressions:

```cpp
// Define a callable from a lambda expression
Callable add_one = [](Float x) { return x + 1.f; };

// A callable may invoke another callable
Callable add_two = [&add_one](Float x) {
    add_one(add_one(x));
};

// A callable may use captured device resources or resources in the argument list
auto buffer = device.create_buffer<float>(...);
Callable copy = [&buffer](BufferFloat buffer2, UInt index) {
    auto x = buffer.read(index); // use captured resource
    buffer2.write(index, x);     // use declared resource in the argument list
};

// Define a 1D kernel from a lambda expression
Kernel1D add_one_and_some = [&buffer, &add_one](Float some, BufferFloat out) {
    auto index = dispatch_id().x;    // query thread index in the whole grid with built-in dispatch_id()
    auto x = buffer.read(index);     // use resource through capturing
    auto result = add_one(x) + some; // invoke a callable
    out.write(index, result);        // use resource in the argument list
};
```

> ⚠️ Note that parameters of the definition functions for callables and kernels must be `Var<T>` or `Var<T> &` (or their aliases).

Kernels can be compiled into shaders by the device, in a blocking or asynchronous fashion:
```cpp
auto some_shader    = device.compile(some_kernel);          // blocking
auto another_shader = device.compile_async(another_kernel); // asynchronous, returns std::shared_future<Shader>
```

Most backends supports caching the compiled shaders to accelerate future compilations of the same shader. The cache files are at `<build-folder>/bin/.cache`.

### Backends, Context, Devices and Resources<a name="devices-and-resources"/>

LuisaCompute currently supports 5 backends:
- CUDA
- DirectX
- Metal
- ISPC
- LLVM

More backends might be added in the future. A device backend is implemented as a plug-in, which follows the `luisa-compute-backend-<name>` naming convention and is placed under `<build-folder>/bin`.

The `Context` object is responsible for loading and managing these plug-ins and creating/destroying devices. Users have to pass the executable path (typically, `argv[0]`) or the runtime directory to a context's constructor (so that it's able to locate the plug-ins), and pass the backend name to create the corresponding device object.
```cpp
int main(int argc, char *argv[]) {
    Context context{argv[0]};
    auto device = context.create_device("cuda");
    /* ... */
}
```

> ⚠️ Creating multiple devices inside the same application is allowed. However, the resources are not shared across devices. Visiting one device's resources from another device's commands/shaders would lead to undefined behaviors.


The device object provides methods for backend-specific operations, typicall, creating resources. LuisaCompute supports the following rousource types:

- `Buffer<T>`s, which are linear memory ranges on the device for structured data storage;
- `Image<T>`s and `Volume<T>`s, which are 2D/3D textures of scalars or vectors readable and writable from the shader, possibly with hardware-accelerated caching and format conversion;
- `BindlessArray`s, which provide slots for references to buffers and textures (`Image`s or `Volume`s bound with texture samplers, read-only in the shader), helpful for reducing the overhead and bypassing the limitations of binding shader parameters;
- `Mesh`es and `Accel`s (short for acceleration structures) for high-performance ray intersection tests, with hardware acceleration if available (e.g., on graphics cards that feature RT-Cores);

<img alt="hardware_resources" src="https://user-images.githubusercontent.com/7614925/196001295-a5407f09-77a0-461a-ab23-ab768ddc08e9.jpg" align="center" width="65%"/>


Devices are also responsible for
- Creating `Stream`s and `Event`s (the former are for command submission and the latter are for host-stream and stream-stream synchronization); and
- Compiling kernels into shaders, as introduced before.

Synopsis of the public interfaces in `class Device`:
```cpp
class Device {
/* ... */
public:
    [[nodiscard]] Stream create_stream(bool for_present = false) noexcept;// see definition in runtime/stream.cpp
    [[nodiscard]] Event create_event() noexcept;                          // see definition in runtime/event.cpp

    [[nodiscard]] SwapChain create_swapchain(
        uint64_t window_handle, const Stream &stream, uint2 resolution,
        bool allow_hdr = true, uint back_buffer_count = 1) noexcept;

    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh create_mesh(
        VBuffer &&vertices, TBuffer &&triangles,
        AccelUsageHint hint = AccelUsageHint::FAST_TRACE) noexcept;                             // see definition in rtx/mesh.h
        
    [[nodiscard]] Accel create_accel(AccelUsageHint hint = AccelUsageHint::FAST_TRACE) noexcept;// see definition in rtx/accel.cpp
    [[nodiscard]] BindlessArray create_bindless_array(size_t slots = 65536u) noexcept;          // see definition in runtime/bindless_array.cpp

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u) noexcept {
        return _create<Image<T>>(pixel, make_uint2(width, height), mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u) noexcept {
        return _create<Image<T>>(pixel, size, mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint width, uint height, uint depth, uint mip_levels = 1u) noexcept {
        return _create<Volume<T>>(pixel, make_uint3(width, height, depth), mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u) noexcept {
        return _create<Volume<T>>(pixel, size, mip_levels);
    }

    template<typename T>
    [[nodiscard]] auto create_buffer(size_t size) noexcept {
        return _create<Buffer<T>>(size);
    }

    template<size_t N, typename... Args>
    [[nodiscard]] auto compile(const Kernel<N, Args...> &kernel, luisa::string_view meta_options = {}) noexcept {
        return _create<Shader<N, Args...>>(kernel.function(), meta_options);
    }

    template<size_t N, typename... Args>
    [[nodiscard]] auto compile_async(const Kernel<N, Args...> &kernel, luisa::string_view meta_options = {}) noexcept {
        return ThreadPool::global().async([this, f = kernel.function(), opt = luisa::string{meta_options}] {
            return _create<Shader<N, Args...>>(f, opt);
        });
    }

    // clang-format off
    template<size_t N, typename Func>
        requires std::negation_v<detail::is_dsl_kernel<std::remove_cvref_t<Func>>>
    [[nodiscard]] auto compile(Func &&f, std::string_view meta_options = {}) noexcept {
        if constexpr (N == 1u) {
            return compile(Kernel1D{std::forward<Func>(f)});
        } else if constexpr (N == 2u) {
            return compile(Kernel2D{std::forward<Func>(f)});
        } else if constexpr (N == 3u) {
            return compile(Kernel3D{std::forward<Func>(f)});
        } else {
            static_assert(always_false_v<Func>, "Invalid kernel dimension.");
        }
    }
    template<size_t N, typename Func>
        requires std::negation_v<detail::is_dsl_kernel<std::remove_cvref_t<Func>>>
    [[nodiscard]] auto compile_async(Func &&f, std::string_view meta_options = {}) noexcept {
        if constexpr (N == 1u) {
            return compile_async(Kernel1D{std::forward<Func>(f)});
        } else if constexpr (N == 2u) {
            return compile_async(Kernel2D{std::forward<Func>(f)});
        } else if constexpr (N == 3u) {
            return compile_async(Kernel3D{std::forward<Func>(f)});
        } else {
            static_assert(always_false_v<Func>, "Invalid kernel dimension.");
        }
    }
};
```

All resources, shaders, streams, and events are C++ objects with *move* contrutors/assignments and following the *RAII* idiom, i.e., automatically calling the `Device::destroy_*` interfaces when destructed.

> ⚠️ Users may need to pay attention not to dangle a resource, e.g., accidentally releases it before the dependent commands finish.

### Command Submission and Synchronization

LuisaCompute adopts the explicit command-based execution model. Conceptually, commands are description units of atomic computation tasks, such as transferring data between the device and host, or from one resource to another; building meshes and acceleration structures; populating or updating bindless arrays; and most importantly, launching shaders.

Commands are organized into command buffers and then submitted to streams which are essentially queues forwarding commands to the backend devices in a logically first-in-first-out (FIFO) manner.

The resource wrappers provide convenient methods for creating commands, e.g.,
```cpp
auto buffer_upload_command   = buffer.copy_from(host_data)
auto accel_build_command     = accel.build();
auto shader_dispatch_command = shader(args...).dispatch(n);
```
Command buffers are group commands that are submitted together:
```cpp
auto command_buffer = stream.command_buffer();
command_buffer
    << raytrace_shader(framebuffer, accel, resolution)
        .dispatch(resolution)
    << accumulate_shader(accum_image, framebuffer)
        .dispatch(resolution)
    << hdr2ldr_shader(accum_image, ldr_image)
        .dispatch(resolution)
    << ldr_image.copy_to(host_image.data())
    << commit(); // the commands are submitted to the stream together on commit()
```

For convenience, a stream implicitly creates a proxy object, which submit commands in the internal command buffer at the end of statements:
```cpp
stream << buffer.copy_from(host_data) // a stream proxy is created on Stream::operator<<()
       << accel.build()               // consecutive commands are stored in the implicit commad buffer in the proxy object
       << raytracing(image, accel, i)
           .dispatch(width, height);  // the proxy object automatically submits the commands at the end of the statement
```

> ⚠️ Since commands are asynchronously executed, users should pay attention to resource and host data lifetimes.

The backends in LuisaCompute can automatically determine the dependencies between the commands in a command buffer, and re-schedule them into an optimized order to improve hardware ultilization. Therefore, larger command buffers might be preferred for better computation throughput.

<img alt="command scheduling" src="https://user-images.githubusercontent.com/7614925/196001465-2dace78b-5e3b-4b4b-b2c3-f2cd61adc6ff.jpg" align="center" width="60%"/>

Multiple streams run concurrently. Therefore, users may require synchronizations between them or with respect to the host via `Event`s, similar to condition variables that ensure ordering across threads:
```cpp
auto event = device.create_event();
stream_a << command_a
         << event.signal(); // signals an event
stream_b << event.wait()    // waits until the event signals
         << command_b;      // will be executed after the event signals
         << event.signal(); // signals again
event.synchronize();        // blocks until the event signals
```

## Applications

We implement several proof-of-concept examples in tree under `src/tests` (sorry for the misleading naming; they are also test programs we used during the development). Besides, you may also found the following applications interesting:

- [LuisaRender](https://github.com/LuisaGroup/LuisaRender.git), a high-performance cross-platform Monte Carlo renderer.
- [LuisaShaderToy](https://github.com/LuisaGroup/LuisaShaderToy.git), a collection of amazing shaders ported from [Shadertoy](https://www.shadertoy.com).

## Documentation and Tutorials

Sorry that we are still working on them. Currently, we would recommand reading the original [paper](https://luisa-render.com) and learning through the examples and applications.

If you have any problem or suggestion, please just feel free to open an [issue](https://github.com/LuisaGroup/LuisaCompute/issues) or start a [discussion](https://github.com/LuisaGroup/LuisaCompute/discussions). We are very happy to hear from you!

## Roadmap

See [ROADMAP.md](ROADMAP.md).


## Citation

```bibtex
@article{Zheng2022LuisaRender,
    author = {Zheng, Shaokun and Zhou, Zhiqian and Chen, Xin and Yan, Difei and Zhang, Chuyan and Geng, Yuefeng and Gu, Yan and Xu, Kun},
    title = {LuisaRender: A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures},
    year = {2022},
    issue_date = {December 2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {41},
    number = {6},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3550454.3555463},
    doi = {10.1145/3550454.3555463},
    journal = {ACM Trans. Graph.},
    month = {nov},
    articleno = {232},
    numpages = {19},
    keywords = {stream architecture, rendering framework, cross-platform renderer}
}
```

The [publisher version](https://doi.org/10.1145/3550454.3555463) of the paper should be available soon before SIGGRAPH Asia 2022.
