---
name: lc_native_shader
description: Compile hand-written native compute shaders and dispatch them from a Stream through NativeShaderExt — HLSL on dx/vk (DXC), GLSL on vk (glslang), CUDA C++ on cuda (NVRTC) — and native raster (vertex + pixel) pipelines on dx and vk, covering compile/reflect/load, NativeShaderLauncher binding by index/register+space/positional, uniforms (root constants / push constants / scalar kernel params), the per-argument Usage contract, include dirs, and lifetime.
---

# Native Shader Injection (`NativeShaderExt`)

`NativeShaderExt` (in `include/luisa/backends/ext/native_shader_ext.h`, the canonical
reference) hands native source (HLSL / GLSL / CUDA C++) to the backend's own compiler,
reflects its bindings, creates a backend compute-shader instance, and dispatches it from
a `Stream`. It bypasses the DSL/AST/XIR path entirely.

- **JIT only** (no `ShaderSerializer`/AOT, no `LUISA_DUMP_SOURCE`).
- **Compute and raster.** A compute module is compiled by `compile()` and dispatched
  by `NativeShaderLauncher::build()`; a vertex+pixel pair is compiled by
  `compile_raster()` and drawn by `NativeShaderLauncher::draw()`. The
  native raster route is implemented on **dx (HLSL only)** and **vk (HLSL or GLSL)**;
  cuda inherits a fail-closed default.
- Only buffers and uniform blocks; textures/samplers/accel reflect but are rejected at `load()`.
- The per-argument `Usage` you declare is the command-reorder synchronization contract.
- Destroy a `NativeShader` before the `Device` that created it.

## Backend support matrix

| `NativeShaderLanguage` | dx | vk | cuda |
|---|---|---|---|
| `HLSL` | DXIL via DXC | SPIR-V via DXC | rejected |
| `GLSL` | rejected | SPIR-V via glslang | rejected |
| `CUDA_NVRTC` | rejected | rejected | PTX via NVRTC |

Native raster (`compile_raster` / `load_raster` / `draw`) is implemented by **dx** (HLSL,
DXIL + DXC reflection, with the root signature and the PSO built from that reflection) and
**vk** (HLSL or GLSL, SPIR-V); the base class fails closed on every other backend, so an
unsupported backend costs no code.

Do not confuse it with the *DSL* rasterisation extension: `RasterExt` in
`luisa/backends/ext/raster_ext.hpp` is provided by **dx, vk and metal4** and compiles a
`RasterKernel`'s DSL functions, not native source. Its `require_raster_ext()` helper
fails closed on cuda/hip/fallback.

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

## Raster (vertex + pixel) pipelines — dx and vk

The same shader object model, but the two stages are compiled as a pair and the backend
builds the graphics pipeline at draw time, because the vertex input layout comes from the
draw's `MeshFormat` and the blend/depth/topology state from its `RasterState` (exactly
like a DSL `RasterShader`).

The dialect is per backend: **GLSL on vk** (`layout(location = n)`, a
`layout(push_constant)` block) and **HLSL on dx** (semantics, a
`cbuffer ... : register(b0)` block fed by root 32-bit constants). vk also accepts HLSL,
spelled with `[[vk::binding]]` / `[[vk::location]]` / `[[vk::push_constant]]`; a pair may
mix languages only where both dialects exist (vk).

```cpp
NativeRasterShaderCompileInfo info;
info.vertex.language = NativeShaderLanguage::GLSL;  // the two stages may differ
info.vertex.source = vertex_source; // string_view, must outlive compile_raster()
info.vertex.entry_point = "main"; // "VSMain" for HLSL
info.pixel = /* same shape, the pixel stage */;
info.push_object_id = false; // true: reserve 4 bytes in front of the block
                             // for RasterMesh::object_id
auto result = ext->compile_raster(info);
if (!result.ok()) { /* result.error names the failing stage(s) */ }
auto shader = ext->create_raster_shader(info); // or load_raster(result)

auto launcher = shader.launcher();
launcher.add_uniform(tint).add_uniform(scale); // the uniform block
stream << std::move(launcher).draw(
            std::move(meshes), mesh_format, Viewport{0, 0, w, h},
            raster_state, depth_buffer_or_nullptr, color_target)
       << synchronize();
```

The contract the two stages must satisfy:
- Vertex inputs follow the *attribute type*, not the stream order: position 0, normal 1,
  tangent 2, color 3, uv0..uv3 4..7. On vk that is
  `location = static_cast<uint32_t>(VertexAttributeType)` (`[[vk::location(n)]]` /
  `layout(location = n)`); on dx it is the *semantic* the draw's `MeshFormat` generates —
  `POSITION`, `NORMAL`, `TANGENT`, `COLOR`, `TEXCOORD0..3` at semantic index 0..3
  (`float4 position : POSITION`).
- The vertex stage writes `SV_Position`/`gl_Position`; pixel outputs are
  `SV_Target0..7` (or `SV_Depth`, for a depth-only draw with no colour target).
  Varyings between the stages are yours, the backend does not rewrite
  them: on vk give both sides matching explicit locations, on dx declare them in the same
  *order* in both stages, because dx links the stages by hardware register (which DXC
  assigns in declaration order — list `SV_Position` first in the pixel stage's input
  struct). `compile_raster()` reports a mismatched register instead of leaving it to the
  pipeline creation. Semantic names are matched case-insensitively, so `COLOR` and
  `color` name the same varying.
- Both stages share one pipeline layout: their binding tables are merged into one in
  canonical order (`NativeRasterShaderCompileResult::bindings`, the launcher's binding
  order), and a `(space, register)` (vk: `(set, binding)`) present in both must have the
  same resource kind. Note that DirectX register namespaces collide in that table, so
  `register(t0)` and `register(u0)` in one pair are a conflict: use another register.
- `push_constant_size` is the *application's* uniform bytes. With `push_object_id` the
  block must start with a 32-bit member for the mesh's object id and the application's
  values start at offset 4 (`{ uint object_id; float shade; }`).
- The uniform block is a push-constant block on vk and a `cbuffer ... : register(b0)` fed
  by root 32-bit constants on dx (at most 64 dwords, object id included); on dx the
  declared size must cover the bytes the shader *reads* from that block.
- A draw's `MeshFormat` must outlive the command, like the DSL raster path.

`NativeRasterShader` is an RAII owner; `destroy_raster_shader()` is the raw form. A native
raster shader is built in memory on both backends, so unlike the DSL raster path it needs
no AOT round-trip (the serializer is not involved at all).

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
- Raster: `build()` on a raster launcher and `draw()` on a compute launcher both assert —
  they would bind the wrong pipeline type. Use `NativeRasterShader::launcher()` + `draw()`.
- Raster: `push_constant_size` smaller than the block a stage declares is rejected (the
  shader would read past the range the launcher fills); a *larger* value is legal, because
  a Vulkan pipeline layout may declare a wider range than a stage's block.
- Raster: two stages that declare the same `(set, binding)` with different resource kinds
  are rejected — one pipeline layout cannot satisfy both.
- Raster: an HLSL `ConstantBuffer` push block is padded to a 16-byte register, so the
  reflected size is >= the application's byte count; leave `push_constant_size = 0` on the
  HLSL stages and let the reflection decide.
- Raster on dx: the two stages must declare their varyings in the same order. DirectX links
  them by hardware register (DXC numbers them in declaration order), so a pixel stage that
  reads `COLOR` while the vertex stage writes `SV_Position` first is a pipeline-creation
  failure. `compile_raster()` names the offending varying and both registers.
