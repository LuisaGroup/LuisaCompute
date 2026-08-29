# Project Architecture

Coroutine suspension points may carry versioned, typed extensions for
scheduling, graph stages, and debugging. Their ownership, IR lifetime, fallback
policy, and cache rules are specified in
[Coroutine suspend extensions](coro_suspend_extensions.md).

This document describes the internal architecture of LuisaCompute, including the compilation pipeline, runtime system, and backend implementations.

## Overview

LuisaCompute is structured in three main layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│         (User code, kernels, resource management)           │
├─────────────────────────────────────────────────────────────┤
│                      Frontend (DSL)                         │
│    (C++ template metaprogramming, AST construction)         │
├─────────────────────────────────────────────────────────────┤
│                    Middle-end (Runtime)                     │
│   (Resource wrappers, command encoding, device interface)   │
├─────────────────────────────────────────────────────────────┤
│              Backend (Platform-specific)                    │
│ (CUDA, DirectX, Metal, Vulkan, HIP, fallback execution)     │
└─────────────────────────────────────────────────────────────┘
```

## Frontend: The Embedded DSL

The Domain Specific Language (DSL) layer allows users to write GPU kernels in C++.

### AST Construction

When you write a kernel:

```cpp
Kernel2D my_kernel = [&](ImageFloat img) noexcept {
    Var coord = dispatch_id().xy();
    img->write(coord, make_float4(1.0f));
};
```

The following happens at C++ runtime:

1. **FunctionBuilder Activation**: A `FunctionBuilder` singleton is pushed to a thread-local stack
2. **Argument Creation**: DSL variables (`Var<T>`) are created as AST expression nodes
3. **Operator Overloading**: Each operation (`+`, `*`, function calls) records AST nodes
4. **AST Finalization**: The builder is popped and the function is finalized

### Key Components

#### FunctionBuilder

The `FunctionBuilder` class in `include/luisa/ast/function_builder.h` is the core of AST construction:

```cpp
// Simplified concept
class FunctionBuilder {
    // Records expressions
    const Expression* literal(const Type* type, const void* data);
    const Expression* binary(BinaryOp op, const Expression* lhs, const Expression* rhs);
    const Expression* call(const Function* func, luisa::span<const Expression* const> args);
    
    // Records statements
    void assign(const Expression* lhs, const Expression* rhs);
    void if_(const Expression* cond, const Statement* true_branch, const Statement* false_branch);
    void for_(const Statement* init, const Expression* cond, const Expression* step, const Statement* body);
};
```

#### Var<T> and Expression

`Var<T>` wraps an AST expression pointer:

```cpp
template<typename T>
struct Var : public Ref<T> {
    Var() : Ref<T>{FunctionBuilder::current()->local(Type::of<T>())} {}
    // Operations delegate to FunctionBuilder
};
```

When you write `Float a = b + c`, it roughly translates to:

```cpp
// a = b + c becomes:
auto expr = FunctionBuilder::current()->binary(
    BinaryOp::ADD, 
    b.expression(), 
    c.expression()
);
FunctionBuilder::current()->assign(a.expression(), expr);
```

### Type System

The type system supports:

- **Scalar types**: `bool`, `int`, `uint`, `float`, `short`, `ushort`, `slong`, `ulong`, `half`
- **Vector types**: `Vector<T, N>` for N = 2, 3, 4
- **Matrix types**: `Matrix<N>` for N = 2, 3, 4
- **Arrays**: `std::array<T, N>`
- **Structures**: User-defined with `LUISA_STRUCT` macro

Type reflection is implemented in `include/luisa/ast/type.h`:

```cpp
// Type registry ensures unique type instances
const Type* type = Type::of<float3>();  // Returns singleton
```

## Middle-end: Runtime and Resources

The runtime layer provides a unified interface over different GPU APIs.

### DeviceInterface

`DeviceInterface` (in `include/luisa/runtime/rhi/device_interface.h`) is the abstraction over backends:

```cpp
class DeviceInterface {
public:
    // Resource creation
    virtual ResourceCreationInfo create_buffer(size_t size) = 0;
    virtual ResourceCreationInfo create_image(...) = 0;
    
    // Shader compilation
    virtual uint64_t create_shader(...) = 0;
    
    // Command execution
    virtual void dispatch(Stream* stream, const CommandList& commands) = 0;
    
    // Synchronization
    virtual void synchronize_event(uint64_t event) = 0;
};
```

### Resource Management

Resources follow a handle-based design:

```cpp
// Resource is a lightweight wrapper
template<typename T>
class Resource {
    DeviceInterface* _device;
    uint64_t _handle;
public:
    // RAII: destructor calls _device->destroy_resource(_handle)
    ~Resource() { _device->destroy_resource(_handle); }
};
```

The actual GPU memory is managed by the backend implementation.

### Command System

Commands are encoded as lightweight description objects:

```cpp
// Command hierarchy
class Command {
    virtual void accept(CommandVisitor& visitor) = 0;
};

class ShaderDispatchCommand : public Command {
    uint64_t shader_handle;
    luisa::vector<Argument> arguments;
    uint3 dispatch_size;
};

class BufferUploadCommand : public Command {
    uint64_t buffer_handle;
    void* host_data;
    size_t offset;
    size_t size;
};
```

### Command Scheduling

A key feature is automatic dependency tracking:

```cpp
// Commands are analyzed for resource usage
void CommandReorderVisitor::visit(const ShaderDispatchCommand& cmd) {
    for (auto& arg : cmd.arguments) {
        if (arg.type == Argument::BUFFER) {
            auto usage = cmd.get_resource_usage(arg.buffer);
            // Track read-after-write, write-after-read dependencies
            _dependencies.emplace_back(arg.buffer, usage, cmd);
        }
    }
}
```

The scheduler builds a DAG of commands and can:
- Reorder independent commands for better throughput
- Insert necessary memory barriers
- Batch compatible commands

## Backend: Code Generation and Execution

Each backend translates the AST to platform-specific code.

### CUDA Backend

**File**: `src/backends/cuda/`

The CUDA backend:
1. Generates PTX or CUDA C++ from the AST
2. Uses NVCC or NVRTC for compilation
3. Manages `CUmodule`, `CUfunction`, and `CUdeviceptr`

```cpp
// Simplified code generation
void CUDACodegenAST::visit(const BinaryExpr* expr) {
    emit("(");
    expr->lhs()->accept(*this);
    emit(" %s ", binary_op_name(expr->op()));
    expr->rhs()->accept(*this);
    emit(")");
}
```

### DirectX Backend

**File**: `src/backends/dx/`

The DirectX backend:
1. Generates HLSL from the AST
2. Uses DXC for compilation to DXIL
3. Manages `ID3D12PipelineState`, `ID3D12Resource`

Key features:
- Root signature generation for descriptor binding
- Resource barrier tracking for state transitions

### Metal Backend

**File**: `src/backends/metal/`

The Metal backend:
1. Generates Metal Shading Language from the AST
2. Uses Metal compiler framework
3. Manages `MTLLibrary`, `MTLFunction`, `MTLBuffer`

### Fallback Backend

**File**: `src/backends/fallback/`

The native C++ fallback backend uses:
- LLVM for code generation
- Embree for ray tracing
- Custom threading for parallel execution

## IR (Intermediate Representation)

XIR is LuisaCompute's native C++ intermediate representation for analysis and optimization.

### AST to IR Conversion

```
AST (FunctionBuilder) -> XIR (luisa::compute::xir) -> Backend Code
```

The IR provides:
- SSA (Static Single Assignment) form for easier analysis
- Explicit control flow graph
- Type-preserving transformations

### IR Passes

Located in `include/luisa/xir/passes/`:

- **DCE (Dead Code Elimination)**: Removes unused computations
- **Mem2Reg**: Promotes stack variables to registers
- **Reg2Mem**: Converts registers back to memory for complex control flow
- **Outline**: Extracts code into separate functions
- **Autodiff**: Automatic differentiation transformation

```cpp
// Example: Mem2Reg pass
void mem2reg_pass(Function* func) {
    // Analyze alloca sites
    // Promote to registers where possible
    // Insert phi nodes at merge points
}
```

## Shader Compilation Pipeline

The full compilation flow:

```
1. User defines Kernel/Callable in C++
   ↓
2. Lambda executes, records AST via FunctionBuilder
   ↓
3. AST is finalized into a Function object
   ↓
4. device.compile(kernel) is called
   ↓
5. Backend converts AST to platform source (PTX, HLSL, MSL, LLVM IR)
   ↓
6. Platform compiler generates machine code
   ↓
7. Backend creates Shader object with compiled binary
   ↓
8. Shader is ready for dispatch
```

### Shader Caching

Compiled shaders are cached to disk:

```
<build-dir>/bin/.cache/
├── cuda/
│   └── <hash>.ptx
├── dx/
│   └── <hash>.dxil
└── metal/
    └── <hash>.metallib
```

Cache key includes:
- AST hash
- Backend version
- Compilation options (fast math, debug info, etc.)

## Memory Management

### GPU Memory

Backends use different allocation strategies:

| Backend | Allocator | Strategy |
|---------|-----------|----------|
| CUDA | cuMemAlloc/cuMemPool | Memory pools for efficiency |
| DirectX | D3D12MA | TLSF-based custom allocator |
| Metal | MTLHeap | Heap-based sub-allocation |
| Fallback | mimalloc | Host-side allocation |

### Host Memory

Pinned memory for efficient transfers:

```cpp
// Pinned memory (page-locked) for fast GPU upload
void* pinned = allocate_pinned(size);
cudaMemcpyAsync(gpu_ptr, pinned, size, cudaMemcpyHostToDevice, stream);
```

## Threading Model

### Host-Side

- **Stream execution**: Commands are queued and executed asynchronously
- **Callback system**: Host functions can be scheduled to run after GPU work
- **Fiber support**: Integration with marl for coroutine-style programming

### Device-Side

Kernels execute in a 3D grid:

```
Grid (dispatch_size)
├── Block (block_size)
│   ├── Warp (SIMD width, e.g., 32 threads)
│   │   └── Threads execute in lockstep
│   └── Multiple warps per block
└── Multiple blocks per grid
```

## Extension System

Backends can expose platform-specific features:

```cpp
// Extension interface
struct DenoiserExt : DeviceExtension {
    static constexpr string_view name = "DenoiserExt";
    virtual void denoise(...) = 0;
};

// Backend implementation
class CUDADenoiser : public DenoiserExt {
    void denoise(...) override;
};

// Usage
if (auto* denoiser = device.extension<DenoiserExt>()) {
    denoiser->denoise(...);
}
```

## Build System Integration

### CMake Integration

LuisaCompute uses modern CMake (3.23+) with the following configuration options:

#### Main Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `LUISA_COMPUTE_ENABLE_CUDA` | ON | Enable CUDA backend |
| `LUISA_COMPUTE_ENABLE_DX` | ON | Enable DirectX backend |
| `LUISA_COMPUTE_ENABLE_METAL` | ON | Enable Metal backend |
| `LUISA_COMPUTE_ENABLE_FALLBACK` | Developer builds | Enable native C++ LLVM/Embree fallback backend |
| `LUISA_COMPUTE_ENABLE_VULKAN` | ON | Enable Vulkan backend |
| `LUISA_COMPUTE_ENABLE_HIP` | OFF | Enable HIP backend (WIP) |
| `LUISA_COMPUTE_ENABLE_DSL` | ON | Enable C++ DSL |
| `LUISA_COMPUTE_ENABLE_GUI` | ON | Enable GUI support |
| `LUISA_COMPUTE_ENABLE_TENSOR` | OFF | Enable tensor extension |
| `LUISA_COMPUTE_BUILD_TESTS` | ON | Build test suite |

#### Build Commands

```bash
# Basic build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# With specific backends
cmake -S . -B build \
    -DLUISA_COMPUTE_ENABLE_CUDA=ON \
    -DLUISA_COMPUTE_ENABLE_DX=OFF \
    -DLUISA_COMPUTE_ENABLE_METAL=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Using Ninja (recommended)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

#### Using LuisaCompute in Your Project

**Method 1: add_subdirectory**

```bash
# Clone with submodules
git clone --recursive https://github.com/LuisaGroup/LuisaCompute.git

# In your CMakeLists.txt, add:
# add_subdirectory(LuisaCompute)
# target_link_libraries(your_target PRIVATE luisa::compute)
```

**Method 2: FetchContent**

```cmake
include(FetchContent)
FetchContent_Declare(
    luisacompute
    GIT_REPOSITORY https://github.com/LuisaGroup/LuisaCompute.git
    GIT_TAG next
    GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(luisacompute)

target_link_libraries(your_target PRIVATE luisa::compute)
```

**Method 3: find_package (after installation)**

```bash
# Install LuisaCompute first
cmake --build build --target install

# In your CMakeLists.txt
find_package(LuisaCompute REQUIRED)
target_link_libraries(your_target PRIVATE LuisaCompute::compute)
```

### XMake Integration

LuisaCompute uses XMake (3.0.6+) as an alternative build system with a more streamlined workflow.

#### Main Build Options

| Option | Default | Description |
|--------|---------|-------------|
| **Backend Options** |||
| `lc_cuda_backend` | true | Enable NVIDIA CUDA backend |
| `lc_dx_backend` | true | Enable DirectX 12 backend |
| `lc_vk_backend` | true | Enable Vulkan backend |
| `lc_metal_backend` | true | Enable Metal backend |
| `lc_fallback_backend` | false | Enable fallback backend |
| **Backend Extensions** |||
| `lc_cuda_ext_lcub` | false | Enable NVIDIA CUB extension (long compile time) |
| `lc_dx_cuda_interop` | false | Enable DirectX-CUDA interop |
| `lc_vk_cuda_interop` | false | Enable Vulkan-CUDA interop |
| **Module Options** |||
| `lc_enable_dsl` | true | Enable C++ DSL module |
| `lc_enable_gui` | true | Enable GUI module |
| `lc_enable_imgui` | true | Enable ImGui support |
| `lc_enable_osl` | true | Enable OSL (Open Shading Language) support |
| `lc_enable_py` | true | Enable Python bindings |
| `lc_enable_clangcxx` | false | Enable Clang C++ module |
| `lc_enable_xir` | true | Enable XIR |
| **Build Configuration** |||
| `lc_enable_mimalloc` | true | Use mimalloc as default allocator |
| `lc_enable_custom_malloc` | false | Enable custom malloc |
| `lc_enable_unity_build` | true | Enable unity (jumbo) build for faster compilation |
| `lc_enable_simd` | true | Enable SSE and SSE2 SIMD |
| `lc_use_lto` | false | Enable Link Time Optimization |
| `lc_rtti` | false | Enable C++ RTTI |
| `lc_cxx_standard` | cxx20 | C++ standard (cxx20, cxx23, etc.) |
| `lc_c_standard` | clatest | C standard |
| `lc_enable_tests` | true | Enable test suite |
| `lc_external_marl` | false | Use external marl library |
| `lc_use_system_stl` | false | Use system STL instead of EASTL |
| **Python Configuration** |||
| `lc_py_include` | false | Python include path |
| `lc_py_linkdir` | false | Python library directory |
| `lc_py_libs` | false | Python libraries to link |
| **Path Configuration** |||
| `lc_bin_dir` | bin | Custom binary output directory |
| `lc_sdk_dir` | false | Custom SDK directory |
| `lc_llvm_path` | false | LLVM installation path (for fallback backend) |
| `lc_embree_path` | false | Embree path (for fallback ray tracing) |
| `lc_toolchain` | false | Custom toolchain |
| `lc_win_runtime` | false | Windows runtime library |
| `lc_optimize` | false | Additional optimization flags |
| **Third-Party Source** |||
| `lc_spdlog_use_xrepo` | false | Use xrepo for spdlog |
| `lc_reproc_use_xrepo` | false | Use xrepo for reproc |
| `lc_lmdb_use_xrepo` | false | Use xrepo for lmdb |
| `lc_imgui_use_xrepo` | false | Use xrepo for imgui |
| `lc_glfw_use_xrepo` | false | Use xrepo for glfw |
| `lc_yyjson_use_xrepo` | false | Use xrepo for yyjson |

#### Build Commands

```bash
# Basic(Release) build
xmake f -m release -c
xmake

# With specific backends
xmake f -m release --lc_cuda_backend=true --lc_dx_backend=false --lc_metal_backend=false -c
xmake

# Debug build
xmake f -m debug -c
xmake

# Using ClangCL toolchain (recommended on Windows)
xmake f -m release --toolchain=clang-cl -c
xmake
```

#### Local Configuration with options.lua

You can create `scripts/options.lua` to save default configuration for your local environment:

```bash
# Generate default options.lua
xmake lua scripts/write_options.lua
```

Example `scripts/options.lua`:

```lua
lc_options = {
    toolchain = "clang-cl",           -- Use LLVM clang-cl compiler
    lc_enable_tests = true,           -- Enable test-case compilation
    lc_enable_gui = false             -- Disable GUI targets
}
```

Options in `options.lua` can be overridden by command-line arguments:

```bash
xmake f --lc_enable_tests=false -c
```

#### Using LuisaCompute in Your Project

**Method 1: Git Submodule + includes**

```bash
# Clone with submodules
git submodule add https://github.com/LuisaGroup/LuisaCompute.git third_party/LuisaCompute
git submodule update --init --recursive
```

```lua
-- Include LuisaCompute's build scripts
includes("third_party/LuisaCompute")

target("your_app")
    set_kind("binary")
    add_deps("lc-dsl")
```

**Method 2: External Project with xmake.repo**

```lua
-- xmake.lua for your project
set_languages("c++20")

-- Require LuisaCompute as a package
add_requires("luisa-compute")

target("your_app")
    set_kind("binary")
    add_files("src/*.cpp")
    add_packages("luisa-compute")
```

### Integration Best Practices

Based on [LuisaRender](https://github.com/LuisaGroup/LuisaRender) and [LuisaComputeGaussianSplatting](https://github.com/LuisaGroup/LuisaComputeGaussianSplatting):

1. **Always use submodules** to ensure consistent versions:
   ```bash
   git submodule add https://github.com/LuisaGroup/LuisaCompute.git third_party/LuisaCompute
   git submodule update --init --recursive
   ```

2. **Set RPATH for portable binaries**:
   ```cmake
   set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
   set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
   ```

3. **Handle backends gracefully**:
   ```cmake
   if(LUISA_COMPUTE_ENABLE_CUDA AND CUDA_FOUND)
       target_compile_definitions(your_target PRIVATE ENABLE_CUDA)
   endif()
   ```

4. **Use unity build for faster compilation:**
   ```bash
   xmake f --lc_enable_unity_build=true -c
   ```
   
5. **Use `-c` flag for clean configuration when switching options:**
   ```bash
   xmake f --lc_cuda_backend=false --lc_vk_backend=true -c
   ```

## Performance Considerations

### Kernel Optimization

1. **Memory coalescing**: Ensure threads access consecutive memory
2. **Occupancy**: Balance register usage and block size
3. **Branch divergence**: Minimize divergent execution within warps
4. **Texture caching**: Use images for 2D spatial locality

### Runtime Optimization

1. **Command batching**: Submit multiple commands at once
2. **Resource reuse**: Avoid repeated allocations
3. **Async transfers**: Overlap compute and data transfer
4. **Stream parallelism**: Use multiple streams for independent work

## Debugging and Profiling

### Validation Layer

Enabled via `LUISA_ENABLE_VALIDATION=1`:

- Resource lifetime tracking
- Memory access validation
- Command buffer consistency checks

### Profiling

Backend-specific profiling:

```cpp
// CUDA: Nsight Systems/Compute integration
// DirectX: PIX markers
// Metal: Xcode GPU debugger
```

## Future Directions

Planned architectural improvements:

1. **Graph-based execution**: Explicit compute graphs for better optimization
2. **Multi-device support**: Seamless multi-GPU scaling
3. **Task graph API**: Higher-level task description
4. **JIT specialization**: Runtime kernel specialization based on parameters
