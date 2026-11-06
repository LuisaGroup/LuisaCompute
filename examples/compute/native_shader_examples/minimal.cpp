// The smallest complete native shader examples: one compute kernel and one
// vertex+pixel rasterisation pair, each compiled from hand-written native source,
// submitted through a `Stream`, and verified on the host.
//
// Run as: example_native_shader_minimal <backend> [compute|raster] [hlsl|glsl]
//         backend: dx | vk | cuda        mode: compute (default) | raster
//         language: hlsl | glsl          (defaults to what the backend speaks)
//
// Native language support: HLSL on dx and vk, GLSL on vk, CUDA C++ on cuda.
// `compute` runs on dx, vk and cuda today. `raster` needs the *native* raster
// route, which dx (HLSL only) and vk (HLSL and GLSL) implement; a backend that does
// not inherits the fail-closed default and reports that.
// `RasterExt`, the DSL rasterisation path, is a different feature: it exists on dx
// and vk, but it compiles DSL functions rather than native source.
//
// Both modes are deliberately independent of the JSON-driven
// `example_native_shader` next to this file: no document, no serializer, no GUI.
// Everything an application has to do is here, in order:
//
//   1. get the extension:   device.extension<NativeShaderExt>()
//   2. compile the source:  ext->compile(info)      (compute)
//                           ext->compile_raster(..) (raster: both stages)
//   3. create the instance: ext->load(result) / ext->load_raster(result)
//   4. build the command:   shader.launcher()...build(thread_count)   (compute)
//                           shader.launcher()...draw(...)             (raster)
//   5. enqueue it:          stream << cmd << synchronize()
//
// The kernels are written per backend because each speaks a different native
// language - that is the structural difference from a DSL kernel, where the
// source is generated from C++ for every backend.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;

namespace {

// ===========================================================================
// compute mode: dst[i] = src[i] * k + c over `element_count` floats, 64 threads
// per workgroup
// ===========================================================================

// DirectX: HLSL with the classic register namespaces. `register(b0)` is the
// uniform block, fed by `add_uniform` (root 32-bit constants).
constexpr auto hlsl_dx = R"(
StructuredBuffer<float> src : register(t0);
RWStructuredBuffer<float> dst : register(u0);
cbuffer Uniforms : register(b0) {
    float k;
    float c;
};
[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = src[tid.x] * k + c;
}
)";

// Vulkan: HLSL, but with the SPIR-V facing attributes, because the module's
// decorations are the authoritative binding table. Plain registers work too
// (`t0` -> binding 0); the explicit form is what the reflection reports.
constexpr auto hlsl_vk = R"(
struct Uniforms { float k; float c; };
[[vk::binding(0, 0)]] StructuredBuffer<float> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dst;
[[vk::push_constant]] ConstantBuffer<Uniforms> uniforms;
[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = src[tid.x] * uniforms.k + uniforms.c;
}
)";

// Vulkan: the same kernel in GLSL, which needs no DXC at all. `readonly` maps to
// a read-only binding, the plain `buffer` to a writable one, and
// `layout(push_constant)` to the uniform block.
constexpr auto glsl_vk = R"(
#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer Src { float values[]; } src;
layout(set = 0, binding = 1) buffer Dst { float values[]; } dst;
layout(push_constant) uniform Push {
    float k;
    float c;
} uniforms;
void main() {
    uint i = gl_GlobalInvocationID.x;
    dst.values[i] = src.values[i] * uniforms.k + uniforms.c;
}
)";

// CUDA: on this route the kernel signature *is* the binding declaration. The
// pointer parameters become the buffer bindings in order (`const float *` is
// read-only), the non-pointer ones become the scalar kernel parameters fed by
// `add_uniform`, and `__launch_bounds__` gives the workgroup size that the other
// routes read from `[numthreads]`.
constexpr auto cuda_source = R"(
extern "C" __global__ void __launch_bounds__(64)
scale(const float *src, float *dst, float k, float c) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i] * k + c;
}
)";

constexpr size_t element_count = 1024u;
constexpr uint compute_block_size = 64u;

// ===========================================================================
// raster mode: two quads with disjoint footprints written to two render targets
// ===========================================================================
//
// A raster "pass" is described entirely by data the application supplies, and the
// shader pair only has to agree with it:
//
//   * vertex input layout  -> `MeshFormat`: which attributes exist, their pixel
//                             formats, and therefore their byte offsets
//   * fixed-function state -> `RasterState`: cull mode, topology, blend, depth
//   * attachments          -> one `Image`/`ImageView` per render target, plus an
//                             optional `DepthBuffer`
//   * viewport             -> `Viewport`
//
// Vertex inputs are bound by *semantic* location, not by stream order:
// `VertexAttributeType::Position` is location 0 and `Color` is location 3. The
// vertex stage writes `gl_Position`/`SV_Position`; the pixel stage writes one
// output per bound render target (`layout(location = 0..1)` /
// `SV_Target0..1`).
//
// The pair is written twice, once per native language - GLSL (vk) and HLSL (dx and
// vk). The host code below is identical for all of them; only the source, the
// language tag and the entry-point name change.
constexpr auto raster_vertex_glsl = R"(
#version 450
layout(location = 0) in vec4 in_position;   // VertexAttributeType::Position
layout(location = 3) in vec4 in_color;      // VertexAttributeType::Color
layout(location = 0) out vec4 v_color;
layout(push_constant) uniform Push {
    vec4 tint;
} uniforms;
void main() {
    gl_Position = in_position;
    v_color = in_color * uniforms.tint;
}
)";

// Two outputs: the tinted colour, and its "complement" in the second render
// target - so a probe can tell a missing second attachment from a wrong one.
constexpr auto raster_pixel_glsl = R"(
#version 450
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o_aux;
void main() {
    o_color = v_color;
    o_aux = vec4(1.0 - v_color.rgb, 1.0);
}
)";

// The same pair in HLSL for Vulkan: explicit `[[vk::location]]` bindings and a
// `[[vk::push_constant]]` block, because the SPIR-V decorations are the
// authoritative binding table.
constexpr auto raster_vertex_hlsl_vk = R"(
struct VSInput {
    [[vk::location(0)]] float4 position : POSITION;
    [[vk::location(3)]] float4 color : COLOR;
};
struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR;
};
struct Push {
    float4 tint;
};
[[vk::push_constant]] ConstantBuffer<Push> uniforms;
VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = input.position;
    output.color = input.color * uniforms.tint;
    return output;
}
)";

constexpr auto raster_pixel_hlsl_vk = R"(
struct PSInput {
    [[vk::location(0)]] float4 color : COLOR;
};
struct PSOutput {
    float4 color : SV_Target0;
    float4 aux : SV_Target1;
};
PSOutput PSMain(PSInput input) {
    PSOutput output;
    output.color = input.color;
    output.aux = float4(1.0 - input.color.rgb, 1.0);
    return output;
}
)";

// And in HLSL for DirectX: semantics instead of `[[vk::...]]`, and `register(b0)`
// as the uniform block (fed by root 32-bit constants). The pixel stage lists
// `SV_Position` first because DirectX links the two stages by *hardware register*:
// the vertex stage's `SV_Position` takes register 0, so `COLOR` is register 1 on
// both sides only when the pixel stage declares its interpolators in the same
// order - the DirectX counterpart of an explicit `[[vk::location]]`.
constexpr auto raster_vertex_hlsl_dx = R"(
struct VSInput {
    float4 position : POSITION;
    float4 color : COLOR;
};
struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
cbuffer Push : register(b0) {
    float4 tint;
};
VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = input.position;
    output.color = input.color * tint;
    return output;
}
)";

constexpr auto raster_pixel_hlsl_dx = R"(
struct PSInput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
struct PSOutput {
    float4 color : SV_Target0;
    float4 aux : SV_Target1;
};
PSOutput PSMain(PSInput input) {
    PSOutput output;
    output.color = input.color;
    output.aux = float4(1.0 - input.color.rgb, 1.0);
    return output;
}
)";

struct RasterVertex {
    float4 position;
    float4 color;
};

constexpr uint raster_size = 64u;

bool run_compute(Device &device) noexcept {

    // `backend_name()` is a view; own it so it can be printed with %s.
    luisa::string backend{device.backend_name()};
    auto ext = device.extension<NativeShaderExt>();
    if (ext == nullptr) {
        std::printf("[skip] backend '%s' has no NativeShaderExt\n", backend.c_str());
        return true;
    }

    // ---- 1. what to compile ---------------------------------------------
    NativeShaderCompileInfo info;
    if (backend == "dx") {
        info.language = NativeShaderLanguage::HLSL;
        info.source = hlsl_dx;
        info.entry_point = "CSMain";
        info.push_constant_size = 2u * sizeof(float);
    } else if (backend == "vk") {
        // GLSL keeps this example independent of the DXC availability of the
        // build; swap in `hlsl_vk` to exercise the HLSL -> SPIR-V route instead.
        info.language = NativeShaderLanguage::GLSL;
        info.source = glsl_vk;
        info.entry_point = "main";
        info.push_constant_size = 2u * sizeof(float);
    } else if (backend == "cuda") {
        info.language = NativeShaderLanguage::CUDA_NVRTC;
        info.source = cuda_source;
        info.entry_point = "scale";
        info.block_size = make_uint3(compute_block_size, 1u, 1u);
        info.push_constant_size = 2u * sizeof(float);
    } else {
        std::printf("[skip] this example has no native kernel for backend '%s'\n",
                    backend.c_str());
        return true;
    }

    // ---- 2. compile ------------------------------------------------------
    auto compiled = ext->compile(info);
    if (!compiled.ok()) {
        std::printf("[error] compile failed: %s\n", compiled.error.c_str());
        return false;
    }
    std::printf("compiled a native %s shader: %zu byte(s), workgroup (%u %u %u), "
                "%zu reflected binding(s)\n",
                backend.c_str(), compiled.binary.size(),
                compiled.block_size.x, compiled.block_size.y, compiled.block_size.z,
                compiled.bindings.size());
    for (auto &&binding : compiled.bindings) {
        std::printf("  binding set %u register %u: kind %d, usage 0x%x\n",
                    binding.space_index, binding.register_index,
                    static_cast<int>(binding.kind),
                    static_cast<unsigned>(luisa::to_underlying(binding.usage)));
    }

    // ---- 3. create the instance -----------------------------------------
    // `NativeShader` is the RAII owner: it destroys the instance on scope exit,
    // which must happen after the dispatch has been synchronized (below).
    auto metadata = ext->load(compiled);
    if (!metadata.valid()) {
        std::printf("[error] load failed: %s\n", compiled.error.c_str());
        return false;
    }
    NativeShader shader{*ext, std::move(metadata)};

    // ---- 4. dispatch -----------------------------------------------------
    constexpr auto k = 2.5f;
    constexpr auto c = -1.5f;
    Stream stream = device.create_stream(StreamTag::COMPUTE);
    Buffer<float> src = device.create_buffer<float>(element_count);
    Buffer<float> dst = device.create_buffer<float>(element_count);

    luisa::vector<float> host_src(element_count);
    for (auto i = 0u; i < element_count; i++) {
        host_src[i] = static_cast<float>(i) * 0.5f - 3.0f;
    }
    stream << src.copy_from(luisa::span{host_src});

    auto launcher = shader.launcher();
    launcher.add_buffer_by_index(0u, src.view(), Usage::READ)
        .add_buffer_by_index(1u, dst.view(), Usage::WRITE)
        .add_uniform(k)
        .add_uniform(c);
    // `build` takes the *total thread count*, not a grid size; the block size
    // comes from the shader (reflected `[numthreads]` / `local_size_*`).
    auto command = std::move(launcher).build(make_uint3(element_count, 1u, 1u));
    stream << std::move(command) << synchronize();

    // ---- 5. verify -------------------------------------------------------
    luisa::vector<float> host_dst(element_count);
    stream << dst.copy_to(luisa::span{host_dst}) << synchronize();

    auto worst = 0.0f;
    for (auto i = 0u; i < element_count; i++) {
        auto expected = host_src[i] * k + c;
        worst = std::max(worst, std::abs(host_dst[i] - expected));
        if (host_dst[i] != expected) {
            std::printf("[error] dst[%u] = %f, expected %f\n",
                        i, host_dst[i], expected);
            return false;
        }
    }
    std::printf("%zu element(s) verified, max |error| = %g\n", element_count, worst);
    std::printf("PASSED compute on '%s'\n", backend.c_str());
    return true;
}

bool run_raster(Device &device, luisa::string_view language) noexcept {

    luisa::string backend{device.backend_name()};
    auto ext = device.extension<NativeShaderExt>();
    auto raster_ext = device.extension<RasterExt>();
    if (ext == nullptr) {
        std::printf("[skip] backend '%s' has no NativeShaderExt\n", backend.c_str());
        return true;
    }
    if (raster_ext == nullptr) {
        // The draw path itself (MeshFormat/RasterState/attachments, clear commands)
        // comes from RasterExt; a backend without it cannot be drawn to.
        std::printf("[skip] backend '%s' has no RasterExt, which the draw needs\n",
                    backend.c_str());
        return true;
    }

    // ---- 1./2. compile the vertex + pixel pair ---------------------------
    // Which source the backend speaks: HLSL on dx and vk, GLSL on vk only. The
    // pair's two stages share one pipeline layout and one push-constant block, so
    // they must agree on it; `push_constant_size` is the application's part of the
    // block (a reserved object-id slot, when requested, sits in front of it).
    NativeRasterShaderCompileInfo info;
    auto set_language = [&](NativeShaderLanguage lang, luisa::string_view vs,
                            luisa::string_view ps, luisa::string_view vs_entry,
                            luisa::string_view ps_entry) noexcept {
        info.vertex.language = lang;
        info.vertex.source = vs;
        info.vertex.entry_point = vs_entry;
        info.vertex.push_constant_size = sizeof(float4);
        info.pixel.language = lang;
        info.pixel.source = ps;
        info.pixel.entry_point = ps_entry;
        info.pixel.push_constant_size = sizeof(float4);
    };
    if (language == "glsl") {
        if (backend != "vk") {
            std::printf("[skip] GLSL native stages are a vk-only route; '%s' speaks HLSL\n",
                        backend.c_str());
            return true;
        }
        set_language(NativeShaderLanguage::GLSL, raster_vertex_glsl, raster_pixel_glsl,
                     "main", "main");
    } else if (language == "hlsl") {
        if (backend == "vk") {
            set_language(NativeShaderLanguage::HLSL, raster_vertex_hlsl_vk,
                         raster_pixel_hlsl_vk, "VSMain", "PSMain");
        } else if (backend == "dx") {
            set_language(NativeShaderLanguage::HLSL, raster_vertex_hlsl_dx,
                         raster_pixel_hlsl_dx, "VSMain", "PSMain");
        } else {
            std::printf("[skip] this example has no HLSL raster pair for backend '%s'\n",
                        backend.c_str());
            return true;
        }
    } else {
        std::printf("[error] unknown language '%s'; expected 'hlsl' or 'glsl'\n",
                    luisa::string{language}.c_str());
        return false;
    }
    auto compiled = ext->compile_raster(info);
    if (!compiled.ok()) {
        // dx and vk implement the native raster route; any other backend that
        // reaches this point does not, so its failure is a skip rather than an
        // error.
        auto has_raster_route = backend == "dx" || backend == "vk";
        std::printf(has_raster_route ? "[error] raster compile failed: %s\n" : "[skip] the native raster pair did not compile on '%s': %s\n",
                    backend.c_str(), compiled.error.c_str());
        return !has_raster_route;
    }
    std::printf("compiled a native raster pair (%s): vertex %zu byte(s), pixel %zu byte(s), "
                "%zu merged binding(s), %u uniform byte(s), object-id slot %u byte(s)\n",
                luisa::string{language}.c_str(),
                compiled.vertex.binary.size(), compiled.pixel.binary.size(),
                compiled.bindings.size(), compiled.push_constant_size,
                compiled.object_id_bytes);

    // ---- 3. create the pipeline -----------------------------------------
    auto metadata = ext->load_raster(compiled);
    if (!metadata.valid()) {
        std::printf("[error] load_raster failed\n");
        return false;
    }
    NativeRasterShader shader{*ext, std::move(metadata)};

    // ---- 4. the pass description: mesh format, state, targets -----------
    // Vertex input layout. The attribute *types* become the vertex-input
    // locations, so the shader's `location 0/3` lines up with the stream layout.
    MeshFormat mesh_format;
    {
        VertexAttribute attributes[] = {
            {VertexAttributeType::Position, PixelFormat::RGBA32F},
            {VertexAttributeType::Color, PixelFormat::RGBA32F},
        };
        mesh_format.emplace_vertex_stream(attributes);
    }
    // Two quads with a gap between them, so the probes can tell "drawn" from
    // "cleared" as well as "left" from "right". Clip space has y = -1 at the top
    // of the viewport.
    constexpr auto left_edge = -0.25f;
    constexpr auto right_edge = 0.25f;
    std::array<RasterVertex, 12u> vertices{
        RasterVertex{make_float4(-1.f, -1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(left_edge, -1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(left_edge, 1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(-1.f, -1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(left_edge, 1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(-1.f, 1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(right_edge, -1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
        RasterVertex{make_float4(1.f, -1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
        RasterVertex{make_float4(1.f, 1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
        RasterVertex{make_float4(right_edge, -1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
        RasterVertex{make_float4(1.f, 1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
        RasterVertex{make_float4(right_edge, 1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
    };
    Buffer<RasterVertex> vertex_buffer = device.create_buffer<RasterVertex>(vertices.size());
    VertexBufferView vertex_view{vertex_buffer};
    // Fixed-function state: no culling (clip-space winding is easy to get wrong),
    // and a depth test that the cleared buffer accepts.
    constexpr auto state = RasterState{
        .cull_mode = CullMode::None,
        .depth_state = {.enable_depth = true,
                        .comparison = Comparison::Less,
                        .write = true}};
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    // Two render targets and one depth buffer.
    Image<float> color_target = device.create_image<float>(
        PixelStorage::FLOAT4, raster_size, raster_size, 1u, false, true);
    Image<float> aux_target = device.create_image<float>(
        PixelStorage::FLOAT4, raster_size, raster_size, 1u, false, true);
    DepthBuffer depth = device.create_depth_buffer(DepthFormat::D32,
                                                   uint2{raster_size, raster_size});
    stream << vertex_buffer.copy_from(luisa::span{vertices});

    luisa::vector<RasterMesh> meshes;
    // The object id only reaches the shader when the pair asked for the reserved
    // 4-byte push-constant slot (`NativeRasterShaderCompileInfo::push_object_id`);
    // it is per mesh, which is why it is a `RasterMesh` field.
    meshes.emplace_back(luisa::span<VertexBufferView const>{&vertex_view, 1u},
                        static_cast<uint>(vertices.size()), 1u, /*object_id*/ 7u);
    auto launcher = shader.launcher();
    launcher.add_uniform(make_float4(0.5f, 1.f, 1.f, 1.f));// `tint`
    auto clear = make_float4(0.1f, 0.1f, 0.1f, 1.f);
    stream << raster_ext->clear_render_target(color_target.view(), clear)
           << raster_ext->clear_render_target(aux_target.view(), clear)
           << depth.clear(1.f)
           // `mesh_format` must outlive the command: the command keeps a pointer
           // to it, exactly like the DSL raster path.
           << std::move(launcher).draw(std::move(meshes), mesh_format,
                                       Viewport{0u, 0u, raster_size, raster_size},
                                       state, &depth, color_target, aux_target)
           << synchronize();

    // ---- 5. verify both targets ------------------------------------------
    luisa::vector<float> color_pixels(raster_size * raster_size * 4u);
    luisa::vector<float> aux_pixels(raster_size * raster_size * 4u);
    stream << color_target.copy_to(luisa::span{color_pixels})
           << aux_target.copy_to(luisa::span{aux_pixels})
           << synchronize();
    auto at = [](luisa::vector<float> const &pixels, uint x, uint y) noexcept {
        auto i = (y * raster_size + x) * 4u;
        return make_float4(pixels[i], pixels[i + 1u], pixels[i + 2u], pixels[i + 3u]);
    };
    auto check = [](float4 actual, float4 expected, const char *what) noexcept {
        constexpr auto eps = 1e-4f;
        auto ok = std::abs(actual.x - expected.x) <= eps &&
                  std::abs(actual.y - expected.y) <= eps &&
                  std::abs(actual.z - expected.z) <= eps &&
                  std::abs(actual.w - expected.w) <= eps;
        std::printf("  %-26s (%g, %g, %g, %g) %s\n", what,
                    actual.x, actual.y, actual.z, actual.w, ok ? "ok" : "MISMATCH");
        return ok;
    };
    auto lx = raster_size / 8u;
    auto rx = 7u * raster_size / 8u;
    auto my = raster_size / 2u;
    auto ok = true;
    // Target 0: the tinted colour of each quad, and the clear color in the gap.
    ok = check(at(color_pixels, lx, my), make_float4(0.5f, 0.f, 0.f, 1.f),
               "target0 left quad (tinted)") &&
         ok;
    ok = check(at(color_pixels, rx, my), make_float4(0.f, 0.f, 1.f, 1.f),
               "target0 right quad") &&
         ok;
    ok = check(at(color_pixels, raster_size / 2u, my), clear,
               "target0 clear color in the gap") &&
         ok;
    // Target 1: the complement written by the same pixel stage, which is what
    // proves the second render target is actually bound.
    ok = check(at(aux_pixels, lx, my), make_float4(0.5f, 1.f, 1.f, 1.f),
               "target1 left quad (complement)") &&
         ok;
    ok = check(at(aux_pixels, rx, my), make_float4(1.f, 1.f, 0.f, 1.f),
               "target1 right quad (complement)") &&
         ok;
    ok = check(at(aux_pixels, raster_size / 2u, my), clear,
               "target1 clear color in the gap") &&
         ok;
    std::printf("%s raster (%s) on '%s'\n", ok ? "PASSED" : "FAILED",
                luisa::string{language}.c_str(), backend.c_str());
    return ok;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        std::printf("usage: %s <backend> [compute|raster] [hlsl|glsl]\n", argv[0]);
        std::printf("       backend: dx | vk | cuda\n");
        std::printf("       compute: HLSL on dx and vk, GLSL on vk, CUDA C++ on cuda\n");
        std::printf("       raster:  HLSL on dx and vk, GLSL on vk\n");
        return 1;
    }
    luisa::string_view mode = argc > 2 ? luisa::string_view{argv[2]} : "compute";
    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto succeeded = false;
    if (mode == "compute") {
        succeeded = run_compute(device);
    } else if (mode == "raster") {
        // Default to the language the backend speaks natively; vk can do both.
        luisa::string_view language = argc > 3 ?
                                          luisa::string_view{argv[3]} :
                                          (luisa::string_view{device.backend_name()} == "dx" ?
                                               luisa::string_view{"hlsl"} :
                                               luisa::string_view{"glsl"});
        succeeded = run_raster(device, language);
    } else {
        std::printf("[error] unknown mode '%s'; expected 'compute' or 'raster'\n",
                    argv[2]);
        std::fflush(stdout);
        return 1;
    }
    // Flush before the device is torn down: the backend's teardown does not
    // reach the CRT's exit flush, so buffered `stdout` would be lost. An
    // application that logs through `LUISA_INFO` does not need this.
    std::fflush(stdout);
    return succeeded ? 0 : 1;
}
