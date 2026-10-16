---
name: lc_native_shader
description: Compile hand-written native compute shaders and dispatch them from a Stream through NativeShaderExt — HLSL on dx/vk (DXC), GLSL on vk (glslang), CUDA C++ on cuda (NVRTC) — covering compile/reflect/load, NativeShaderLauncher binding by index/register+space/positional, uniforms (root constants / push constants / scalar kernel params), the per-argument Usage contract, include dirs, and lifetime.
---

# Native Shader Injection (`NativeShaderExt`)

`NativeShaderExt` (in `include/luisa/backends/ext/native_shader_ext.h`, the canonical
reference) hands native source (HLSL / GLSL / CUDA C++) to the backend's own compiler,
reflects its bindings, creates a backend compute-shader instance, and dispatches it from
a `Stream`. It bypasses the DSL/AST/XIR path entirely.

- **Compute only**, **JIT only** (no `ShaderSerializer`/AOT, no `LUISA_DUMP_SOURCE`).
- Only buffers and uniform blocks; textures/samplers/accel reflect but are rejected at `load()`.
- The per-argument `Usage` you declare is the command-reorder synchronization contract.
- Destroy a `NativeShader` before the `Device` that created it.

## Backend support matrix

| `NativeShaderLanguage` | dx | vk | cuda |
|---|---|---|---|
| `HLSL` | DXIL via DXC | SPIR-V via DXC | rejected |
| `GLSL` | rejected | SPIR-V via glslang | rejected |
| `CUDA_NVRTC` | rejected | rejected | PTX via NVRTC |

`compile()` fails closed (`ok()` false, `binary` empty, `error` names the reason).

## Workflow

```cpp
#include <luisa/backends/ext/native_shader_ext.h>
// ... runtime/context.h, runtime/device.h, runtime/stream.h, runtime/command_list.h ...

auto ext = device.extension<NativeShaderExt>();
if (ext == nullptr) { /* backend has no NativeShaderExt */ }

// 1. compile source -> bytecode + reflection
NativeShaderCompileInfo info;
info.language           = NativeShaderLanguage::HLSL;  // or GLSL / CUDA_NVRTC
info.source             = source;                      // string_view: must outlive compile()
info.entry_point        = "CSMain";                    // "main" for GLSL
info.push_constant_size = 2u * sizeof(float);          // uniform bytes; 0 = reflect
// info.block_size for CUDA when the kernel has no __launch_bounds__
// info.source_type = NativeShaderSourceType::FilePath; info.include_dirs = {...}
// info.shader_model, info.optimize, info.enable_fast_math, info.enable_debug_info
auto result = ext->compile(info);
if (!result.ok()) { /* result.error */ }

// 2. load bytecode -> backend shader instance + resolved binding table
auto metadata = ext->load(result);                     // optional usage_override span
if (!metadata.valid()) { /* contract violation, LUISA_WARNING names it */ }

// 3. RAII owner (destroys the instance before the device)
NativeShader shader{*ext, std::move(metadata)};
// or one-shot: auto shader = ext->create_shader(info);

// 4. bind + dispatch
auto launcher = shader.launcher();                     // handle + block size + bindings
launcher.add_buffer_by_index(0u, src.view(), Usage::READ)
        .add_buffer_by_index(1u, dst.view(), Usage::WRITE)
        .add_uniform(k)
        .add_uniform(c);
auto cmd = std::move(launcher).build(uint3{thread_count, 1u, 1u});
stream << std::move(cmd) << synchronize();
```

`build(thread_count)` takes the **total thread count** (grid × block), not a grid size;
the block size comes from the launcher (the shader's reflected workgroup size). Use
`launcher.validate()` for a soft error string; `build()` asserts the same plan.

## Shader sources

**HLSL on dx** — free entry-point name; `[numthreads]` drives the reflected block size:

```hlsl
StructuredBuffer<float> src : register(t0);            // SRV
RWStructuredBuffer<float> dst : register(u0);          // UAV
cbuffer Uniforms : register(b0) { float k; float c; }; // uniform block (root constants)
[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) { dst[tid.x] = src[tid.x] * k + c; }
```
The `b0` cbuffer is the launcher's uniform block, fed by `add_uniform` values and removed
from `load()`'s binding table (`compile()` still reports it). Root-constant limit: 64 ×
32-bit; `push_constant_size` must be a multiple of 4.

**HLSL on vk** — SPIR-V decorations are the authoritative reflection table:

```hlsl
struct Uniforms { float k; float c; };
[[vk::binding(0, 0)]] StructuredBuffer<float> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dst;
[[vk::push_constant]] ConstantBuffer<Uniforms> uniforms;
[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) { dst[tid.x] = src[tid.x] * uniforms.k + uniforms.c; }
```
Plain registers also work (`t<n>`/`u<n>` → binding `n`, space 0); use
`[[vk::binding(set, binding)]]` for a specific set/binding.

**GLSL on vk** — single entry point named `main`:

```glsl
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; } src;   // SRV
layout(set = 0, binding = 1) buffer B { float b[]; } dst;            // UAV
layout(push_constant) uniform Push { float k; float c; } uniforms;
void main() { uint i = gl_GlobalInvocationID.x; dst.b[i] = src.a[i] * uniforms.k + uniforms.c; }
```
`readonly` → `StructuredBuffer`, otherwise `RWStructuredBuffer`. `layout(push_constant)` is
the uniform block (size read back from the module when `push_constant_size == 0`).
Push-constant size must be a multiple of 4. For `#include`, add
`#extension GL_GOOGLE_include_directive : require` after `#version`.

**CUDA C++ on cuda** — the kernel signature *is* the binding declaration:

```cpp
extern "C" __global__ void scale(const float *src, float *dst, float k, float c) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i] * k + c;
}
```
- `extern "C"` keeps the name unmangled; `__global__` must not be templated/qualified.
- Pointer params are buffer bindings in order, each bound as the buffer's 64-bit device
  address: `const T *` → `StructuredBuffer`/`READ`; `T *` → `RWStructuredBuffer`/`READ_WRITE`.
- Non-pointer params are scalar kernel params fed by `add_uniform` in order; there is no
  push-constant block, "uniform bytes" is just their total size (`push_constant_size`
  cross-checks, 0 = reflect it). A by-value struct is valid (`add_uniform` uses `sizeof`).
- `entry_point` selects the `__global__` (default `main` = "the only kernel").
- `block_size` must be supplied unless the kernel declares `__launch_bounds__(N)`.

## Reflection & binding

`result.bindings` / `shader.bindings()` are reported in canonical order
`(space_index, register_index, kind)`; the launcher's positional overload and reorder order
follow it too.

```cpp
struct NativeShaderResourceBinding {
    NativeShaderResourceKind kind;   // StructuredBuffer, RWStructuredBuffer, ConstantBuffer, ...
    uint register_index;             // HLSL register / GLSL binding
    uint space_index;                // HLSL space / GLSL set
    uint array_size;                 // 1 == non-array
    Usage usage;                     // effective usage
    uint32_t stride;                 // element stride when known
    uint32_t size_bytes;             // constant-buffer / block size when known
};
```

Supported kinds at `load()`: **dx** ConstantBuffer, StructuredBuffer/RWStructuredBuffer,
ByteAddressBuffer/RWByteAddressBuffer; **vk** the same plus TypedBuffer/RWTypedBuffer;
**cuda** buffers only. Default usages: CBV/UBO and SRV classes → `READ`; UAV classes →
`READ_WRITE`.

The three bind overloads (prefer index when HLSL namespaces collide — `register(t0)` and
`register(b0)` are both bind point 0):

- `add_buffer_by_index(index, view, usage)` — index into the reflection table (recommended).
- `add_buffer(register, space, view, usage)` — usage disambiguates a shared (register, space).
- `add_buffer(view, usage)` — positional, canonical order.

Every reflected binding must be supplied exactly once; missing/duplicate/extra args and null
handles are rejected by `validate()`/`build()`.

## Launcher & dispatch

```cpp
template<typename T> NativeShaderLauncher &add_uniform(const T &value);
NativeShaderLauncher &add_uniform(const void *data, size_t size, size_t alignment);
auto cmd = std::move(launcher).build(uint3{thread_count, 1u, 1u});
// or: std::move(launcher).build(shader_handle, thread_count, block_size);
```
Uniforms are packed into a blob (offsets shifted past the argument header; alignment
clamped to ≤ 16). `NativeShaderDispatchCommand` is a `CustomDispatchCommand`
(`CustomCommandUUID::NATIVE_SHADER_DISPATCH`, 0x0600) tagged `StreamTag::COMPUTE`; enqueue
it like any other command.

## Usage contract

The declared per-argument usage **is** the synchronization contract: `READ` (mergeable with
other readers), `WRITE` (exclusive), `READ_WRITE` (both); `NONE` is never valid. `build()`
(and the backend at `load()`) cross-check it against the reflected class:

- An SRV/CBV bound as `WRITE` is always rejected.
- A UAV bound as `READ` is rejected by default; allow it with
  `launcher.set_allow_usage_override(true)`.

Declaring `READ` for a resource the shader writes silently breaks synchronization.
`load(result, usage_override)` can override usages up-front (one entry per reflected
binding; a `const T *` binding cannot be declared writable).

## Include dirs & file sources

`NativeShaderCompileInfo` carries `include_dirs`; a `FilePath` source also resolves relative
to its own directory:

```cpp
info.source_type  = NativeShaderSourceType::FilePath;
info.source       = luisa::string_view{source_path};   // must outlive compile()
info.include_dirs = { scratch_dir };
```
Includes go through DXC / glslang's includer / NVRTC -I respectively. A bogus include
directory fails closed on every route: the standalone `luisa_nvrtc` child prints the
compiler log to its stderr, exits non-zero (it never aborts), and the parent surfaces
the log plus the exit code in `result.error`.

## Lifetime

```cpp
{
    NativeShader shader{*ext, std::move(metadata)};   // RAII; move-only
    ...
}   // destroy_shader() here
// or: ext->destroy_shader(handle); / shader.reset();
```
Synchronize dispatches before destruction. A leftover instance is released at extension
teardown with a warning, never silently.

## Pitfalls

- `info.source` / `info.entry_point` are `string_view` — the memory must outlive `compile()`.
- `build(thread_count)` is a thread count, not a grid size (1024 threads @ block 64 →
  `build(uint3{1024,1,1})`).
- On dx, bind by the `load()`/`metadata.bindings` table (`compile()` reports the raw DXC
  table including the b0 cbuffer).
- Block size must match the reflected workgroup size, or `build()` asserts. dx/vk derive it
  from `[numthreads]`/`local_size_*`; cuda requires it (or `__launch_bounds__`).
- UAV-as-`READ` needs `set_allow_usage_override(true)`; SRV-as-`WRITE` is never allowed.
- Destroy the shader before the device and before its dispatches are torn down.
