// Device test for native raster shader injection: a hand-written vertex + pixel
// shader pair, compiled to the backend's bytecode, drawn through the raster (draw)
// command path, and read back for pixel-exact verification.
//
// Run as: test_native_shader_raster <backend>
//         LUISA_TEST_VALIDATION=1 test_native_shader_raster <backend>
//
// The suite is written against two *source dialects* rather than one backend:
//   * GLSL (Vulkan): locations are explicit on both sides
//     (`layout(location = ...)`) and the uniform block is a push-constant block;
//   * HLSL for DirectX: the input layout comes from the *semantics*
//     (POSITION/COLOR at semantic index 0, which is what
//     `RasterShader::get_mesh_format_state` builds) and the uniform block is a
//     `cbuffer ... : register(b0)`, which the root signature feeds with root 32-bit
//     constants. `[[vk::location]]` / `[[vk::push_constant]]` are Vulkan spellings,
//     hence the second HLSL set below.
// Which set runs is decided by capability, not by `backend_name()`: the validation
// layer wraps the device and its name is not the backend's, so a name-based gate
// would silently skip the whole suite there.
//
// Covered risks:
//  * the stage-interaction contract of `NativeRasterShaderCompileInfo`: vertex
//    inputs are bound by *semantic* (location 0 for position, 3 for color), the
//    vertex stage writes `SV_Position`/`gl_Position`, the pixel stage writes
//    `SV_Target0`;
//  * the launcher's uniform blob lands in the root-constant / push-constant block
//    the shader declares (vec4 then float), and a second draw does not inherit the
//    first one's constants;
//  * a real resource binding: the merged binding table and the per-draw
//    root-CBV/SRV/UAV (dx) / descriptor set (vk) binding, and - because the command
//    then carries a non-empty declared-usage list - the native branch of the
//    command-reorder pass and of the barrier recording;
//  * `NativeRasterShaderCompileInfo::push_object_id` is written per mesh of a
//    multi-mesh draw, not once per draw;
//  * depth testing against the DSL `DepthBuffer`, in both draw orders;
//  * indexed and non-indexed draws;
//  * HLSL stages, and - where GLSL is available - a mixed HLSL-vertex/GLSL-pixel
//    pair;
//  * fail-closed diagnostics: a pixel entry point compiled for the vertex stage,
//    an empty source, an unsupported resource class, two stages disagreeing on a
//    binding, a too-small uniform block, and a mismatched usage override.
//
// Every assertion is a pixel probe or a reflected-metadata check, so nothing here
// can pass just because a call returned.
#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <cmath>
#include <cstdlib>

#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
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
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr uint width = 64u;
constexpr uint height = 64u;
// Two 8-bit steps: the quads have hard edges and the depth test is exact, so
// nothing but the 1/255 quantization of a FLOAT4 target moves a value.
constexpr float tolerance = 2.5f / 255.f;

struct RasterVertex {
    float4 position;
    float4 color;
};

// ---- GLSL pair, plain uniforms ------------------------------------------
// `tint` (vec4) then `scale` (float) is exactly what the launcher packs, and
// std430 puts `scale` at byte 16, which is the launcher's packed offset too.
constexpr auto glsl_vertex_source = R"(
#version 450
layout(location = 0) in vec4 in_position;
layout(location = 3) in vec4 in_color;
layout(location = 0) out vec4 v_color;
layout(push_constant) uniform Push {
    vec4 tint;
    float scale;
} u;
void main() {
    gl_Position = in_position;
    v_color = vec4(in_color.rgb * u.tint.rgb * u.scale, in_color.a);
}
)";

constexpr auto glsl_pixel_source = R"(
#version 450
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
void main() { o_color = v_color; }
)";

// ---- GLSL pair, reserved object-id slot ---------------------------------
// The reserved 4 bytes are pushed by the backend before each mesh, so the
// application's own block starts at byte 4.
constexpr auto glsl_object_id_vertex_source = R"(
#version 450
layout(location = 0) in vec4 in_position;
layout(location = 3) in vec4 in_color;
layout(location = 0) out vec4 v_color;
layout(push_constant) uniform Push {
    uint object_id;
    float shade;
} u;
void main() {
    gl_Position = in_position;
    v_color = vec4(in_color.rgb * u.shade * (1.0 + float(u.object_id)), 1.0);
}
)";

constexpr auto glsl_object_id_pixel_source = R"(
#version 450
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
void main() { o_color = v_color; }
)";

// A pixel stage with *two* outputs: the tinted colour and its complement, so a
// probe can tell a missing second attachment from a wrong one.
constexpr auto glsl_two_target_pixel_source = R"(
#version 450
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o_aux;
void main() {
    o_color = v_color;
    o_aux = vec4(1.0 - v_color.rgb, 1.0);
}
)";

// ---- GLSL pair with a real resource binding ---------------------------
// The pixel stage reads a `readonly buffer`, i.e. a storage-buffer descriptor at
// (set 0, binding 0): the only pair here that exercises the descriptor path.
constexpr auto glsl_buffer_vertex_source = R"(
#version 450
layout(location = 0) in vec4 in_position;
layout(location = 3) in vec4 in_color;
layout(location = 0) out vec4 v_color;
void main() {
    gl_Position = in_position;
    v_color = in_color;
}
)";

constexpr auto glsl_buffer_pixel_source = R"(
#version 450
layout(set = 0, binding = 0) readonly buffer Tint { vec4 tint[]; };
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
void main() { o_color = v_color * tint[0]; }
)";

// ---- GLSL pair: the uniform block *and* two resource bindings -----------
// The layout arithmetic of the whole route only ever ran at index 0 in the pairs
// above: this one has a push-constant block *and* two storage buffers, so the
// uniform block is not the only parameter and the binding indices are not zero.
// The two buffers are read in *different* channels, so a swapped binding order
// cannot cancel out.
constexpr auto glsl_mixed_vertex_source = R"(
#version 450
layout(location = 0) in vec4 in_position;
layout(location = 3) in vec4 in_color;
layout(location = 0) out vec4 v_color;
layout(push_constant) uniform Push {
    vec4 tint;
    float scale;
} u;
void main() {
    gl_Position = in_position;
    v_color = vec4(in_color.rgb * u.tint.rgb * u.scale, in_color.a);
}
)";

constexpr auto glsl_mixed_pixel_source = R"(
#version 450
layout(set = 0, binding = 0) readonly buffer Tint { vec4 tint[]; };
layout(set = 0, binding = 1) readonly buffer Shade { vec4 shade[]; };
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
void main() {
    o_color = vec4(v_color.r * tint[0].r, v_color.g,
                   v_color.b * shade[0].b, v_color.a);
}
)";

// A pixel shader that samples a texture, i.e. a resource class the native route
// does not bind yet.
constexpr auto glsl_sampler_pixel_source = R"(
#version 450
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
void main() { o_color = v_color * texture(tex, vec2(0.5)); }
)";

// Two stages that claim the same (set, binding) with different resource kinds:
// one pipeline layout cannot satisfy both.
constexpr auto glsl_conflict_vertex_source = R"(
#version 450
layout(set = 0, binding = 0) readonly buffer Vert { vec4 v[]; };
layout(location = 0) in vec4 in_position;
layout(location = 0) out vec4 v_color;
void main() { gl_Position = in_position; v_color = v[0]; }
)";

constexpr auto glsl_conflict_pixel_source = R"(
#version 450
layout(set = 0, binding = 0) uniform PixelBlock { vec4 p; };
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 o_color;
void main() { o_color = v_color * p; }
)";

// ---- HLSL pair for Vulkan -----------------------------------------------
// The same pipeline in HLSL, with the SPIR-V facing spellings: locations are
// explicit on both sides and the uniform block is the
// `[[vk::push_constant]]` constant buffer.
constexpr auto hlsl_vk_vertex_source = R"(
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
    float scale;
};
[[vk::push_constant]] ConstantBuffer<Push> u;
VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = input.position;
    output.color = float4(input.color.rgb * u.tint.rgb * u.scale, input.color.a);
    return output;
}
)";

constexpr auto hlsl_vk_pixel_source = R"(
struct PSInput {
    [[vk::location(0)]] float4 color : COLOR;
};
float4 PSMain(PSInput input) : SV_Target {
    return input.color;
}
)";

// ---- HLSL pair for DirectX ----------------------------------------------
// The DirectX route takes the vertex input layout from the *semantics* - the
// mesh format's attribute types become POSITION/COLOR at semantic index 0, which
// is what `RasterShader::get_mesh_format_state` emits - and the uniform block is a
// `cbuffer ... : register(b0)`, fed by root 32-bit constants. `float4 tint` at
// offset 0 and `float scale` at offset 16 are the launcher's packed layout, the
// same bytes the GLSL pair reads (the block *allocates* 32 bytes because a cbuffer
// rounds up to a 16-byte register, but only the first 20 are read).
constexpr auto hlsl_dx_vertex_source = R"(
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
    float scale;
};
VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = input.position;
    output.color = float4(input.color.rgb * tint.rgb * scale, input.color.a);
    return output;
}
)";

constexpr auto hlsl_dx_pixel_source = R"(
// `SV_Position` first: DirectX links the two stages by *register*, and the vertex
// stage's `SV_Position` occupies register 0, so its first varying sits at register
// 1 - which the pixel stage only reproduces if it declares the interpolators in the
// same order (this is the DirectX counterpart of an explicit `[[vk::location]]`).
struct PSInput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
float4 PSMain(PSInput input) : SV_Target {
    return input.color;
}
)";

constexpr auto hlsl_dx_object_id_vertex_source = R"(
struct VSInput {
    float4 position : POSITION;
    float4 color : COLOR;
};
struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
cbuffer Push : register(b0) {
    uint object_id;
    float shade;
};
VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = input.position;
    output.color = float4(input.color.rgb * shade * (1.0 + float(object_id)), 1.0);
    return output;
}
)";

constexpr auto hlsl_dx_two_target_pixel_source = R"(
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

// A real StructuredBuffer binding: `register(t0)` becomes a root SRV on dx.
constexpr auto hlsl_dx_buffer_vertex_source = R"(
struct VSInput {
    float4 position : POSITION;
    float4 color : COLOR;
};
struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = input.position;
    output.color = input.color;
    return output;
}
)";

constexpr auto hlsl_dx_buffer_pixel_source = R"(
StructuredBuffer<float4> tint : register(t0);
struct PSInput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
float4 PSMain(PSInput input) : SV_Target {
    return input.color * tint[0];
}
)";

// The same pair in the DirectX dialect: `register(b0)` for the uniform block plus
// `register(t0)`/`register(t1)` for two root SRVs, i.e. a root signature with three
// parameters where the constants are *not* parameter 0.
constexpr auto hlsl_dx_mixed_vertex_source = R"(
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
    float scale;
};
VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = input.position;
    output.color = float4(input.color.rgb * tint.rgb * scale, input.color.a);
    return output;
}
)";

constexpr auto hlsl_dx_mixed_pixel_source = R"(
StructuredBuffer<float4> tint : register(t0);
StructuredBuffer<float4> shade : register(t1);
struct PSInput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
float4 PSMain(PSInput input) : SV_Target {
    return float4(input.color.r * tint[0].r, input.color.g,
                  input.color.b * shade[0].b, input.color.a);
}
)";

// A resource class the route does not bind yet: a sampled image and its sampler.
constexpr auto hlsl_dx_sampler_pixel_source = R"(
Texture2D<float4> tex : register(t0);
SamplerState tex_sampler : register(s0);
struct PSInput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
float4 PSMain(PSInput input) : SV_Target {
    return input.color * tex.Sample(tex_sampler, float2(0.5, 0.5));
}
)";

// A resource declared as an *array* spans more than one register, which this route
// (one root parameter / one descriptor per binding) cannot represent: it has to be
// rejected up front instead of being bound as if it were a single resource.
constexpr auto hlsl_dx_array_pixel_source = R"(
StructuredBuffer<float4> tints[2] : register(t0);
struct PSInput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
float4 PSMain(PSInput input) : SV_Target {
    return input.color * tints[0][0];
}
)";

// A pixel stage that writes only depth: legal for a depth-only pipeline with no
// render target, and the reason the pixel-stage witness is not `SV_Target` alone.
constexpr auto hlsl_dx_depth_only_pixel_source = R"(
struct PSInput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
float PSMain(PSInput input) : SV_Depth {
    return input.position.z;
}
)";

// Two stages that claim the same (space, register) with different resource kinds.
// DirectX register namespaces do not separate them in the binding table, so one of
// the two has to give: this is the fail-closed diagnostic for that case.
constexpr auto hlsl_dx_conflict_vertex_source = R"(
StructuredBuffer<float4> vert : register(t0);
struct VSInput {
    float4 position : POSITION;
};
struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = input.position;
    output.color = vert[0];
    return output;
}
)";

constexpr auto hlsl_dx_conflict_pixel_source = R"(
RWStructuredBuffer<float4> pixel_tint : register(u0);
struct PSInput {
    float4 position : SV_Position;
    float4 color : COLOR;
};
float4 PSMain(PSInput input) : SV_Target {
    return input.color * pixel_tint[0];
}
)";

// A pixel stage that lists its interpolators in the *other* order: without
// `SV_Position` first its only varying sits at hardware register 0, while the vertex
// stage above writes `COLOR` at register 1. DirectX links the stages by register, so
// this pair cannot be built - and `compile_raster()` has to say so instead of
// surfacing a bare pipeline-creation failure.
constexpr auto hlsl_dx_wrong_order_pixel_source = R"(
struct PSInput {
    float4 color : COLOR;
};
float4 PSMain(PSInput input) : SV_Target {
    return input.color;
}
)";

// A *pixel* shader declared under the name the vertex profile is asked for. It
// parses and its entry point exists, so the only thing that can reject it is the
// stage selection itself (`vs_6_5` refuses `SV_Target`) - which is what makes it a
// meaningful test of the per-stage compile route. The text is HLSL, so the same
// source serves both HLSL dialects.
constexpr auto hlsl_wrong_stage_source = R"(
float4 VSMain() : SV_Target { return float4(1.0, 0.0, 0.0, 1.0); }
)";

// A vertex + pixel pair of one dialect, with the uniform bytes the application
// pushes for it (`NativeShaderCompileInfo::push_constant_size`).
struct SourcePair {
    NativeShaderLanguage language;
    const char *vertex;
    const char *pixel;
    uint32_t uniform_size;
};

// One complete dialect: every pair the suite draws with, plus the two negative
// cases. Two of the three instances below are complete - GLSL for Vulkan, HLSL for
// DirectX - because on those backends the whole suite is expected to run in that
// dialect; the Vulkan *HLSL* pair only has to exist for the HLSL-specific tests.
struct SourceSet {
    // tint + scale root constants; the pair every probe uses.
    SourcePair quads;
    // Reserved object-id slot + `shade` (uniform_size counts `shade` only).
    SourcePair object_id;
    // quads' vertex stage with a two-render-target pixel stage.
    SourcePair two_target;
    // One StructuredBuffer binding (dx `register(t0)` / vk `set 0 binding 0`).
    SourcePair buffer;
    // Two bindings *and* the uniform block: the layout index of the constants is
    // not 0, so the launcher has to place them after the bindings.
    SourcePair mixed;
    // Two stages claiming the same (space, register) with different kinds.
    SourcePair conflict;
    // A sampled image, i.e. a class the route rejects.
    const char *sampler_pixel;
};

constexpr SourceSet glsl_sources{
    .quads = {NativeShaderLanguage::GLSL, glsl_vertex_source, glsl_pixel_source, 20u},
    .object_id = {NativeShaderLanguage::GLSL, glsl_object_id_vertex_source,
                  glsl_object_id_pixel_source, 4u},
    .two_target = {NativeShaderLanguage::GLSL, glsl_vertex_source,
                   glsl_two_target_pixel_source, 20u},
    .buffer = {NativeShaderLanguage::GLSL, glsl_buffer_vertex_source,
               glsl_buffer_pixel_source, 0u},
    .mixed = {NativeShaderLanguage::GLSL, glsl_mixed_vertex_source,
              glsl_mixed_pixel_source, 20u},
    .conflict = {NativeShaderLanguage::GLSL, glsl_conflict_vertex_source,
                 glsl_conflict_pixel_source, 0u},
    .sampler_pixel = glsl_sampler_pixel_source,
};

constexpr SourceSet dx_hlsl_sources{
    .quads = {NativeShaderLanguage::HLSL, hlsl_dx_vertex_source,
              hlsl_dx_pixel_source, 20u},
    .object_id = {NativeShaderLanguage::HLSL, hlsl_dx_object_id_vertex_source,
                  hlsl_dx_pixel_source, 4u},
    .two_target = {NativeShaderLanguage::HLSL, hlsl_dx_vertex_source,
                   hlsl_dx_two_target_pixel_source, 20u},
    .buffer = {NativeShaderLanguage::HLSL, hlsl_dx_buffer_vertex_source,
               hlsl_dx_buffer_pixel_source, 0u},
    .mixed = {NativeShaderLanguage::HLSL, hlsl_dx_mixed_vertex_source,
              hlsl_dx_mixed_pixel_source, 20u},
    .conflict = {NativeShaderLanguage::HLSL, hlsl_dx_conflict_vertex_source,
                 hlsl_dx_conflict_pixel_source, 0u},
    .sampler_pixel = hlsl_dx_sampler_pixel_source,
};

// The Vulkan HLSL pair: only the tests that must speak HLSL use it.
constexpr SourcePair vk_hlsl_quads{
    NativeShaderLanguage::HLSL, hlsl_vk_vertex_source, hlsl_vk_pixel_source, 20u};

[[nodiscard]] NativeRasterShaderCompileInfo make_pair(
    SourcePair const &sources, bool push_object_id = false) noexcept {
    NativeRasterShaderCompileInfo info;
    info.vertex.language = sources.language;
    info.vertex.source = sources.vertex;
    info.vertex.entry_point =
        sources.language == NativeShaderLanguage::GLSL ? "main" : "VSMain";
    info.vertex.push_constant_size = sources.uniform_size;
    info.pixel.language = sources.language;
    info.pixel.source = sources.pixel;
    info.pixel.entry_point =
        sources.language == NativeShaderLanguage::GLSL ? "main" : "PSMain";
    info.pixel.push_constant_size = sources.uniform_size;
    info.push_object_id = push_object_id;
    return info;
}

// Declares the application's uniform block for both stages (the merged size is the
// larger of the two, so a per-stage override would not move it).
void set_uniform_size(NativeRasterShaderCompileInfo &pair, uint32_t size) noexcept {
    pair.vertex.push_constant_size = size;
    pair.pixel.push_constant_size = size;
}

struct Frame {
    luisa::vector<float> pixels;
    [[nodiscard]] float4 at(uint x, uint y) const noexcept {
        auto i = (y * width + x) * 4u;
        return make_float4(pixels[i], pixels[i + 1u], pixels[i + 2u], pixels[i + 3u]);
    }
};

[[nodiscard]] bool close(float4 a, float4 b, float eps = tolerance) noexcept {
    return std::abs(a.x - b.x) <= eps && std::abs(a.y - b.y) <= eps &&
           std::abs(a.z - b.z) <= eps && std::abs(a.w - b.w) <= eps;
}

[[nodiscard]] luisa::string to_string(float4 v) noexcept {
    return luisa::format("({}, {}, {}, {})", v.x, v.y, v.z, v.w);
}

void test_native_raster(Device &device) {

    auto ext = device.extension<NativeShaderExt>();
    if (ext == nullptr) {
        LUISA_WARNING("Backend '{}' has no NativeShaderExt; skipping.",
                      device.backend_name());
        return;
    }
    // Decide by *capability*, not by `backend_name()`: the validation layer wraps
    // the device and the extension forwards to the real backend, so the name is not
    // "vk"/"dx" there even though the route works (and must then be exercised, not
    // skipped). A known-good pair of each dialect is the probe, and it requires the
    // reflection the suite depends on: no resource bindings (the uniform block is
    // root/push constants, not a binding) and exactly the uniform block the
    // launcher pushes. That is what tells the two HLSL dialects apart - the DirectX
    // spelling of a non-push-constant `cbuffer` reflects as a *binding* on Vulkan.
    auto probe = [&](SourcePair const &pair) noexcept {
        auto info = make_pair(pair);
        auto result = ext->compile_raster(info);
        return result.ok() && result.bindings.empty() &&
               result.push_constant_size == pair.uniform_size;
    };
    auto glsl_available = probe(glsl_sources.quads);
    auto dx_hlsl_available = probe(dx_hlsl_sources.quads);
    auto vk_hlsl_available = probe(vk_hlsl_quads);
    // The suite needs a *complete* dialect. The two backends that implement the
    // route provide one each: Vulkan speaks GLSL, DirectX speaks the DirectX HLSL
    // dialect. A backend that provides the route but neither dialect would have to
    // extend the source sets below instead of being checked here.
    auto const *primary = glsl_available    ? &glsl_sources :
                          dx_hlsl_available ? &dx_hlsl_sources :
                                              nullptr;
    if (primary == nullptr) {
        // No native raster route: the fail-closed default is what has to hold, so
        // that it cannot silently start handing out a usable shader.
        LUISA_WARNING("Backend '{}' has no native raster route; checking the "
                      "fail-closed default.",
                      device.backend_name());
        "native_raster_unsupported_backend_fails_closed"_test = [&] {
            auto info = make_pair(glsl_sources.quads);
            auto result = ext->compile_raster(info);
            expect(!result.ok());
            expect(!result.error.empty());
            expect(result.vertex.binary.empty() && result.pixel.binary.empty());
            auto shader = ext->create_raster_shader(info);
            expect(!static_cast<bool>(shader));
            // An HLSL pair must be refused the same way, so the check does not
            // depend on GLSL being the dialect the backend speaks.
            auto hlsl_info = make_pair(dx_hlsl_sources.quads);
            auto hlsl_result = ext->compile_raster(hlsl_info);
            expect(!hlsl_result.ok());
            expect(!hlsl_result.error.empty());
            expect(!static_cast<bool>(ext->create_raster_shader(hlsl_info)));
        };
        return;
    }
    // The pair the HLSL-specific tests use, if this backend can compile it at all
    // (a Vulkan build without DXC cannot).
    auto const *hlsl_pair = dx_hlsl_available ? &dx_hlsl_sources.quads :
                            vk_hlsl_available ? &vk_hlsl_quads :
                                                nullptr;

    // The vertex attribute contract: position at semantic 0 and color at semantic
    // 3 - the *semantic* index, not the stream order. On dx the semantic names are
    // POSITION/COLOR, on vk the locations 0/3; declaring them in this order also
    // puts color at byte offset 16 in the vertex buffer.
    MeshFormat mesh_format;
    {
        VertexAttribute attributes[] = {
            {VertexAttributeType::Position, PixelFormat::RGBA32F},
            {VertexAttributeType::Color, PixelFormat::RGBA32F},
        };
        mesh_format.emplace_vertex_stream(attributes);
    }

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Image<float> target = device.create_image<float>(
        PixelStorage::FLOAT4, width, height, 1u, false, true);
    DepthBuffer depth = device.create_depth_buffer(DepthFormat::D32, uint2{width, height});
    auto clear_to_black = device.compile<2>([](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.f, 0.f, 0.f, 1.f));
    });
    auto readback = [&] {
        Frame frame;
        frame.pixels.resize(width * height * 4u);
        stream << target.copy_to(luisa::span{frame.pixels}) << synchronize();
        return frame;
    };
    // A launcher is a builder: `draw()` consumes it, so it is always spelled
    // `std::move(launcher).draw(...)`.
    auto make_draw = [&](NativeRasterShader &shader, luisa::vector<RasterMesh> &&meshes,
                         const RasterState &state, DepthBuffer const *dsv,
                         float4 tint, float scale) {
        auto launcher = shader.launcher();
        launcher.add_uniform(tint).add_uniform(scale);
        return std::move(launcher).draw(
            std::move(meshes), mesh_format, Viewport{0u, 0u, width, height},
            state, dsv, target);
    };

    // Two quads with disjoint footprints, in clip space. The first is the left
    // eighth of x and the second the right eighth; the middle strip stays at the
    // clear color. The quads span the whole vertical range, so the y direction of
    // the clip space (Vulkan's y = -1 is the top of the viewport, DirectX's y = +1
    // is) does not move any probe. Quads (two triangles each) rather than single
    // triangles, so that every probe is far from a rasterization edge and cannot
    // depend on the fill rule.
    constexpr auto no_depth_state = RasterState{
        .cull_mode = CullMode::None,
        .depth_state = {}};
    constexpr auto red_quad_vertices = 6u;
    constexpr auto left_edge = -0.25f;
    constexpr auto right_edge = 0.25f;
    std::array<RasterVertex, 12u> quadrant_vertices{
        // red quad, x in [-1, -0.25]
        RasterVertex{make_float4(-1.f, -1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(left_edge, -1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(left_edge, 1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(-1.f, -1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(left_edge, 1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        RasterVertex{make_float4(-1.f, 1.f, 0.f, 1.f), make_float4(1.f, 0.f, 0.f, 1.f)},
        // blue quad, x in [0.25, 1]
        RasterVertex{make_float4(right_edge, -1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
        RasterVertex{make_float4(1.f, -1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
        RasterVertex{make_float4(1.f, 1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
        RasterVertex{make_float4(right_edge, -1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
        RasterVertex{make_float4(1.f, 1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
        RasterVertex{make_float4(right_edge, 1.f, 0.f, 1.f), make_float4(0.f, 0.f, 1.f, 1.f)},
    };
    Buffer<RasterVertex> quadrant_buffer = device.create_buffer<RasterVertex>(12u);
    VertexBufferView quadrant_view{quadrant_buffer};
    stream << quadrant_buffer.copy_from(luisa::span{quadrant_vertices});
    // Probe columns: inside the red quad, the clear-colored middle strip, and
    // inside the blue quad.
    constexpr auto red_x = width / 8u;
    constexpr auto middle_x = width / 2u;
    constexpr auto blue_x = 7u * width / 8u;
    constexpr auto mid_y = height / 2u;
    // A neutral uniform block value: the new "uniform + bindings" pair multiplies
    // the vertex colour by it, so 1/1/1/1 keeps the quad colours unchanged.
    auto neutral_tint = make_float4(1.f, 1.f, 1.f, 1.f);
    auto quadrant_meshes = [&] {
        luisa::vector<RasterMesh> meshes;
        meshes.emplace_back(luisa::span<VertexBufferView const>{&quadrant_view, 1u},
                            12u, 1u, 0u);
        return meshes;
    };
    auto probe_pixel = [](Frame const &frame, uint x, uint y) noexcept {
        return frame.at(x, y);
    };

    // ---- 1. the primary dialect, non-indexed, both quads ----------------
    "native_raster_quads"_test = [&] {
        auto pair = make_pair(primary->quads);
        // The reflected block is exactly the launcher's packed size, which is
        // what makes the std430 / cbuffer offsets line up.
        auto result = ext->compile_raster(pair);
        expect(result.ok()) << result.error.c_str();
        expect(result.bindings.empty()) << "this pair declares no resources";
        expect(result.push_constant_size == primary->quads.uniform_size);
        expect(result.object_id_bytes == 0u);
        auto shader = ext->create_raster_shader(pair);
        expect(static_cast<bool>(shader));
        // Keep going without crashing when the compile/load of this pair failed:
        // a failed `create_raster_shader` must surface as failed assertions, not
        // as a draw with an invalid handle.
        if (!static_cast<bool>(shader)) { return; }
        expect(shader.object_id_bytes() == 0u);
        expect(shader.bindings().empty()) << "this pair declares no resources";

        stream << clear_to_black(target).dispatch(width, height)
               << make_draw(shader, quadrant_meshes(), no_depth_state, nullptr,
                            make_float4(1.f, 1.f, 1.f, 1.f), 1.f)
               << synchronize();
        auto frame = readback();
        auto red = probe_pixel(frame, red_x, mid_y);
        auto blue = probe_pixel(frame, blue_x, mid_y);
        expect(close(red, make_float4(1.f, 0.f, 0.f, 1.f)))
            << ("the left quad must be the red quad, got " + to_string(red)).c_str();
        expect(close(blue, make_float4(0.f, 0.f, 1.f, 1.f)))
            << ("the right quad must be the blue quad, got " + to_string(blue)).c_str();
        expect(close(probe_pixel(frame, middle_x, mid_y), make_float4(0.f, 0.f, 0.f, 1.f)))
            << "the middle strip must stay at the clear color";
        expect(close(probe_pixel(frame, middle_x, height / 8u), make_float4(0.f, 0.f, 0.f, 1.f)))
            << "the middle strip must stay at the clear color near the top";
    };

    // ---- 2. the uniform block really reaches the vertex stage ------------
    "native_raster_push_constants"_test = [&] {
        auto pair = make_pair(primary->quads);
        auto shader = ext->create_raster_shader(pair);
        expect(static_cast<bool>(shader));
        // Keep going without crashing when the compile/load of this pair failed:
        // a failed `create_raster_shader` must surface as failed assertions, not
        // as a draw with an invalid handle.
        if (!static_cast<bool>(shader)) { return; }
        stream << clear_to_black(target).dispatch(width, height)
               << make_draw(shader, quadrant_meshes(), no_depth_state, nullptr,
                            make_float4(1.f, 1.f, 1.f, 1.f), 0.5f)
               << synchronize();
        auto frame = readback();
        auto red = probe_pixel(frame, red_x, mid_y);
        expect(close(red, make_float4(0.5f, 0.f, 0.f, 1.f)))
            << ("scale must reach the vertex stage, got " + to_string(red)).c_str();
        // A second draw with different constants must not inherit the first
        // one's: the vertex color is red, so only the scale can move the result.
        // A quarter (instead of a half) is neither the previous value nor the
        // clear color, so this also fails if the second draw does not happen.
        stream << clear_to_black(target).dispatch(width, height)
               << make_draw(shader, quadrant_meshes(), no_depth_state, nullptr,
                            make_float4(1.f, 1.f, 1.f, 1.f), 0.25f)
               << synchronize();
        frame = readback();
        red = probe_pixel(frame, red_x, mid_y);
        expect(close(red, make_float4(0.25f, 0.f, 0.f, 1.f)))
            << ("the second draw's constants must replace the first one's, got " +
                to_string(red))
                   .c_str();
    };

    // ---- 3. indexed draw ------------------------------------------------
    "native_raster_indexed_draw"_test = [&] {
        // Only the red quad goes through the index buffer, so the image differs
        // from the non-indexed case: if the indexed draw were replaced by a plain
        // one, the blue quad (vertices 6..11, outside this index range) would
        // appear too.
        std::array<uint, 6u> indices{0u, 1u, 2u, 3u, 4u, 5u};
        Buffer<uint> index_buffer = device.create_buffer<uint>(6u);
        stream << index_buffer.copy_from(luisa::span{indices});
        auto pair = make_pair(primary->quads);
        auto shader = ext->create_raster_shader(pair);
        expect(static_cast<bool>(shader));
        // Keep going without crashing when the compile/load of this pair failed:
        // a failed `create_raster_shader` must surface as failed assertions, not
        // as a draw with an invalid handle.
        if (!static_cast<bool>(shader)) { return; }
        luisa::vector<RasterMesh> meshes;
        meshes.emplace_back(luisa::span<VertexBufferView const>{&quadrant_view, 1u},
                            index_buffer.view(), 1u, 0u);
        auto launcher = shader.launcher();
        launcher.add_uniform(make_float4(1.f, 1.f, 1.f, 1.f)).add_uniform(1.f);
        stream << clear_to_black(target).dispatch(width, height)
               << std::move(launcher).draw(std::move(meshes), mesh_format,
                                           Viewport{0u, 0u, width, height},
                                           no_depth_state, nullptr, target)
               << synchronize();
        auto frame = readback();
        expect(close(probe_pixel(frame, red_x, mid_y), make_float4(1.f, 0.f, 0.f, 1.f)))
            << "indexed draw must cover the red quad";
        expect(close(probe_pixel(frame, blue_x, mid_y), make_float4(0.f, 0.f, 0.f, 1.f)))
            << "the blue quad is outside the index range and must not be drawn";
    };

    // ---- 4. depth test, in both draw orders ------------------------------
    "native_raster_depth_test"_test = [&] {
        // Two oversized triangles at different depths; the near one must win
        // whichever order they are drawn in, because the depth buffer keeps the
        // nearer fragment.
        std::array<RasterVertex, 6u> near_and_far{
            RasterVertex{make_float4(-1.f, -1.f, 0.9f, 1.f), make_float4(0.f, 1.f, 0.f, 1.f)},
            RasterVertex{make_float4(3.f, -1.f, 0.9f, 1.f), make_float4(0.f, 1.f, 0.f, 1.f)},
            RasterVertex{make_float4(-1.f, 3.f, 0.9f, 1.f), make_float4(0.f, 1.f, 0.f, 1.f)},
            RasterVertex{make_float4(-1.f, -1.f, 0.5f, 1.f), make_float4(1.f, 1.f, 0.f, 1.f)},
            RasterVertex{make_float4(3.f, -1.f, 0.5f, 1.f), make_float4(1.f, 1.f, 0.f, 1.f)},
            RasterVertex{make_float4(-1.f, 3.f, 0.5f, 1.f), make_float4(1.f, 1.f, 0.f, 1.f)},
        };
        Buffer<RasterVertex> buffer = device.create_buffer<RasterVertex>(6u);
        stream << buffer.copy_from(luisa::span{near_and_far});
        VertexBufferView view{buffer};
        auto pair = make_pair(primary->quads);
        auto shader = ext->create_raster_shader(pair);
        expect(static_cast<bool>(shader));
        // Keep going without crashing when the compile/load of this pair failed:
        // a failed `create_raster_shader` must surface as failed assertions, not
        // as a draw with an invalid handle.
        if (!static_cast<bool>(shader)) { return; }
        constexpr auto depth_state = RasterState{
            .cull_mode = CullMode::None,
            .depth_state = {.enable_depth = true,
                            .comparison = Comparison::Less,
                            .write = true}};
        auto draw_range = [&](uint first, uint count) {
            luisa::vector<RasterMesh> meshes;
            meshes.emplace_back(luisa::span<VertexBufferView const>{&view, 1u},
                                count, 1u, 0u, static_cast<int>(first));
            auto launcher = shader.launcher();
            launcher.add_uniform(make_float4(1.f, 1.f, 1.f, 1.f)).add_uniform(1.f);
            return std::move(launcher).draw(std::move(meshes), mesh_format,
                                            Viewport{0u, 0u, width, height},
                                            depth_state, &depth, target);
        };
        // Both orders are drawn on purpose: the first one alone would also pass
        // with depth testing disabled (the near triangle is drawn last), so the
        // second one is the discriminating case.
        auto probe_center = [&](const char *what) {
            auto frame = readback();
            auto c = probe_pixel(frame, width / 2u, height / 2u);
            expect(close(c, make_float4(1.f, 1.f, 0.f, 1.f)))
                << (luisa::string{what} +
                    ": the near (yellow) triangle must win, got " + to_string(c))
                       .c_str();
        };
        stream << clear_to_black(target).dispatch(width, height)
               << depth.clear(1.f)
               << draw_range(0u, 3u)// far first
               << draw_range(3u, 3u)// then near
               << synchronize();
        probe_center("far then near");
        stream << clear_to_black(target).dispatch(width, height)
               << depth.clear(1.f)
               << draw_range(3u, 3u)// near first
               << draw_range(0u, 3u)// then far (must be rejected by the test)
               << synchronize();
        probe_center("near then far");
    };

    // ---- 5. reserved object-id slot, per mesh of one draw ---------------
    "native_raster_object_id"_test = [&] {
        auto pair = make_pair(primary->object_id, /*push_object_id*/ true);
        auto result = ext->compile_raster(pair);
        expect(result.ok()) << result.error.c_str();
        expect(result.object_id_bytes == 4u);
        expect(result.push_constant_size == primary->object_id.uniform_size);
        auto shader = ext->create_raster_shader(pair);
        expect(static_cast<bool>(shader));
        // Keep going without crashing when the compile/load of this pair failed:
        // a failed `create_raster_shader` must surface as failed assertions, not
        // as a draw with an invalid handle.
        if (!static_cast<bool>(shader)) { return; }
        expect(shader.object_id_bytes() == 4u);
        // Both quads take the same tiny input color, so only the object id can
        // tell them apart: id 100 -> 0.101, id 200 -> 0.201.
        std::array<RasterVertex, 12u> vertices{quadrant_vertices};
        for (auto &v : vertices) { v.color = make_float4(0.001f, 0.f, 0.f, 1.f); }
        Buffer<RasterVertex> buffer = device.create_buffer<RasterVertex>(12u);
        stream << buffer.copy_from(luisa::span{vertices});
        VertexBufferView view{buffer};
        // One draw, two meshes with different object ids: the backend must write
        // the object id before each mesh, not once per draw.
        luisa::vector<RasterMesh> meshes;
        meshes.emplace_back(luisa::span<VertexBufferView const>{&view, 1u},
                            6u, 1u, 100u, 0);
        meshes.emplace_back(luisa::span<VertexBufferView const>{&view, 1u},
                            6u, 1u, 200u, 6);
        auto launcher = shader.launcher();
        launcher.add_uniform(1.f);// `shade`
        stream << clear_to_black(target).dispatch(width, height)
               << std::move(launcher).draw(std::move(meshes), mesh_format,
                                           Viewport{0u, 0u, width, height},
                                           no_depth_state, nullptr, target)
               << synchronize();
        auto frame = readback();
        auto left = probe_pixel(frame, red_x, mid_y);
        auto right = probe_pixel(frame, blue_x, mid_y);
        expect(close(left, make_float4(0.101f, 0.f, 0.f, 1.f)))
            << ("mesh with object_id 100 must see 100 in its uniform block, got " +
                to_string(left))
                   .c_str();
        expect(close(right, make_float4(0.201f, 0.f, 0.f, 1.f)))
            << ("mesh with object_id 200 must see 200 in its uniform block, got " +
                to_string(right))
                   .c_str();
    };

    // ---- 6. a real resource binding -------------------------------------
    "native_raster_buffer_binding"_test = [&] {
        // The only pair here that binds a resource instead of root/push constants:
        // it exercises the merged binding table, the per-draw root SRV (dx) /
        // descriptor set (vk) binding, and - because the command then carries a
        // non-empty declared-usage list - the native branch of the command-reorder
        // pass and of the barrier recording.
        auto pair = make_pair(primary->buffer);
        auto result = ext->compile_raster(pair);
        expect(result.ok()) << result.error.c_str();
        expect(result.bindings.size() == 1u)
            << luisa::format("expected one merged binding, got {}",
                             result.bindings.size())
                   .c_str();
        if (result.bindings.size() == 1u) {
            auto &&b = result.bindings.front();
            expect(b.kind == NativeShaderResourceKind::StructuredBuffer)
                << "a readonly buffer must reflect as a structured buffer";
            expect(b.space_index == 0u && b.register_index == 0u);
            expect(b.usage == Usage::READ);
        }
        auto shader = ext->create_raster_shader(pair);
        expect(static_cast<bool>(shader));
        // Keep going without crashing when the compile/load of this pair failed:
        // a failed `create_raster_shader` must surface as failed assertions, not
        // as a draw with an invalid handle.
        if (!static_cast<bool>(shader)) { return; }
        expect(shader.bindings().size() == 1u);
        // The pixel stage multiplies the interpolated color by the buffer's first
        // texel, so a half/quarter grey tint yields 0.5/0.25 of the red quad:
        // distinct from each other and from the clear color.
        std::array<float4, 1u> half{make_float4(0.5f, 0.5f, 0.5f, 1.f)};
        std::array<float4, 1u> quarter{make_float4(0.25f, 0.25f, 0.25f, 1.f)};
        Buffer<float4> half_tint = device.create_buffer<float4>(1u);
        Buffer<float4> quarter_tint = device.create_buffer<float4>(1u);
        auto make_tinted_draw = [&](BufferView<float4> tint) {
            luisa::vector<RasterMesh> meshes;
            meshes.emplace_back(luisa::span<VertexBufferView const>{&quadrant_view, 1u},
                                12u, 1u, 0u);
            auto launcher = shader.launcher();
            launcher.add_buffer_by_index(0u, tint, Usage::READ);
            return std::move(launcher).draw(
                std::move(meshes), mesh_format, Viewport{0u, 0u, width, height},
                no_depth_state, nullptr, target);
        };
        stream << half_tint.copy_from(luisa::span{half})
               << quarter_tint.copy_from(luisa::span{quarter})
               << clear_to_black(target).dispatch(width, height)
               << make_tinted_draw(half_tint.view())
               << synchronize();
        auto frame = readback();
        auto red = probe_pixel(frame, red_x, mid_y);
        expect(close(red, make_float4(0.5f, 0.f, 0.f, 1.f)))
            << ("the pixel stage must read the bound tint buffer, got " +
                to_string(red))
                   .c_str();
        // Two draws of the same shader in one command buffer, with two different
        // buffers: the binding must be rewritten per draw.
        stream << clear_to_black(target).dispatch(width, height)
               << make_tinted_draw(half_tint.view())
               << clear_to_black(target).dispatch(width, height)
               << make_tinted_draw(quarter_tint.view())
               << synchronize();
        frame = readback();
        red = probe_pixel(frame, red_x, mid_y);
        expect(close(red, make_float4(0.25f, 0.f, 0.f, 1.f)))
            << ("the second draw must use its own tint buffer, got " +
                to_string(red))
                   .c_str();
        // The declared usage is the reorder contract: a read-only resource bound
        // as WRITE is rejected by the launcher.
        {
            auto launcher = shader.launcher();
            launcher.add_buffer_by_index(0u, half_tint.view(), Usage::WRITE);
            expect(!launcher.validate().empty())
                << "a structured buffer bound as WRITE must be rejected";
        }
    };

    // ---- 7. the uniform block *and* two resource bindings ---------------
    "native_raster_uniform_and_bindings"_test = [&] {
        // Every pair above has either a uniform block or bindings, so the layout
        // arithmetic of the route only ever ran at index 0. This pair has both: the
        // constants sit *after* the two bindings in the generated layout, and the
        // launcher has to write them to that index while binding the two resources
        // at 0 and 1 - on dx three root parameters, on vk a push-constant block plus
        // two storage-buffer descriptors.
        auto pair = make_pair(primary->mixed);
        auto result = ext->compile_raster(pair);
        expect(result.ok()) << result.error.c_str();
        expect(result.bindings.size() == 2u)
            << luisa::format("expected two merged bindings, got {}",
                             result.bindings.size())
                   .c_str();
        expect(result.push_constant_size == primary->mixed.uniform_size);
        if (result.bindings.size() == 2u) {
            expect(result.bindings[0].register_index == 0u &&
                   result.bindings[1].register_index == 1u)
                << "the merged table must keep the two buffers apart";
            expect(result.bindings[0].usage == Usage::READ &&
                   result.bindings[1].usage == Usage::READ);
        }
        auto shader = ext->create_raster_shader(pair);
        expect(static_cast<bool>(shader));
        // Keep going without crashing when the compile/load of this pair failed:
        // a failed `create_raster_shader` must surface as failed assertions, not
        // as a draw with an invalid handle.
        if (!static_cast<bool>(shader)) { return; }
        expect(shader.bindings().size() == 2u);
        std::array<float4, 1u> tint{make_float4(0.5f, 0.5f, 0.5f, 1.f)};
        std::array<float4, 1u> shade{make_float4(0.25f, 0.25f, 0.25f, 1.f)};
        Buffer<float4> tint_buffer = device.create_buffer<float4>(1u);
        Buffer<float4> shade_buffer = device.create_buffer<float4>(1u);
        auto make_mixed_draw = [&](float scale) {
            luisa::vector<RasterMesh> meshes;
            meshes.emplace_back(luisa::span<VertexBufferView const>{&quadrant_view, 1u},
                                12u, 1u, 0u);
            auto launcher = shader.launcher();
            launcher.add_uniform(neutral_tint)
                .add_uniform(scale)
                .add_buffer_by_index(0u, tint_buffer.view(), Usage::READ)
                .add_buffer_by_index(1u, shade_buffer.view(), Usage::READ);
            return std::move(launcher).draw(
                std::move(meshes), mesh_format, Viewport{0u, 0u, width, height},
                no_depth_state, nullptr, target);
        };
        stream << tint_buffer.copy_from(luisa::span{tint})
               << shade_buffer.copy_from(luisa::span{shade})
               << clear_to_black(target).dispatch(width, height)
               << make_mixed_draw(1.f)
               << synchronize();
        auto frame = readback();
        // The vertex stage applies the uniform block, the pixel stage takes red from
        // `tint[0].r` (0.5) and blue from `shade[0].b` (0.25), so the red and blue
        // quads do not just prove that *something* was bound - they prove which
        // binding went where.
        auto red = probe_pixel(frame, red_x, mid_y);
        auto blue = probe_pixel(frame, blue_x, mid_y);
        expect(close(red, make_float4(0.5f, 0.f, 0.f, 1.f)))
            << ("the red quad must read binding 0 (tint) in red, got " +
                to_string(red))
                   .c_str();
        expect(close(blue, make_float4(0.f, 0.f, 0.25f, 1.f)))
            << ("the blue quad must read binding 1 (shade) in blue, got " +
                to_string(blue))
                   .c_str();
        // Halving the uniform halves both quads: the constants reach the layout index
        // that is not a binding.
        stream << clear_to_black(target).dispatch(width, height)
               << make_mixed_draw(0.5f)
               << synchronize();
        frame = readback();
        red = probe_pixel(frame, red_x, mid_y);
        blue = probe_pixel(frame, blue_x, mid_y);
        expect(close(red, make_float4(0.25f, 0.f, 0.f, 1.f)))
            << ("the uniform block must scale the vertex color, got " +
                to_string(red))
                   .c_str();
        expect(close(blue, make_float4(0.f, 0.f, 0.125f, 1.f)))
            << ("the uniform block must scale the vertex color, got " +
                to_string(blue))
                   .c_str();
    };

    // ---- 8. HLSL pair, and a mixed-language pair ------------------------
    "native_raster_hlsl"_test = [&] {
        if (hlsl_pair == nullptr) {
            LUISA_WARNING("Backend '{}' cannot compile an HLSL native raster pair; "
                          "skipping the HLSL checks.",
                          device.backend_name());
            return;
        }
        auto pair = make_pair(*hlsl_pair);
        auto result = ext->compile_raster(pair);
        expect(result.ok()) << result.error.c_str();
        expect(result.vertex.entry_point == luisa::string{"VSMain"})
            << ("got '" + result.vertex.entry_point + "'").c_str();
        expect(result.pixel.entry_point == luisa::string{"PSMain"})
            << ("got '" + result.pixel.entry_point + "'").c_str();
        // DXC pads the uniform block to a 16-byte register on Vulkan (push
        // constants) while DirectX reports only the bytes the shader reads, so both
        // backends must report at least the 20 application bytes.
        expect(result.push_constant_size >= 20u)
            << luisa::format("HLSL uniform size {}", result.push_constant_size).c_str();
        auto shader = ext->create_raster_shader(pair);
        expect(static_cast<bool>(shader));
        // Keep going without crashing when the compile/load of this pair failed:
        // a failed `create_raster_shader` must surface as failed assertions, not
        // as a draw with an invalid handle.
        if (!static_cast<bool>(shader)) { return; }
        stream << clear_to_black(target).dispatch(width, height)
               << make_draw(shader, quadrant_meshes(), no_depth_state, nullptr,
                            make_float4(1.f, 1.f, 1.f, 1.f), 1.f)
               << synchronize();
        auto frame = readback();
        auto red = probe_pixel(frame, red_x, mid_y);
        auto blue = probe_pixel(frame, blue_x, mid_y);
        expect(close(red, make_float4(1.f, 0.f, 0.f, 1.f)))
            << ("HLSL pair must render the same red quad, got " + to_string(red)).c_str();
        expect(close(blue, make_float4(0.f, 0.f, 1.f, 1.f)))
            << ("HLSL pair must render the same blue quad, got " + to_string(blue)).c_str();
        // A mixed pair: HLSL vertex stage, GLSL pixel stage. Both become modules of
        // the backend's own format, so the pipeline does not care - but only a
        // backend that speaks GLSL can be asked for it.
        if (!glsl_available) {
            LUISA_WARNING("Backend '{}' has no GLSL stages; skipping the "
                          "mixed-language pair.",
                          device.backend_name());
            return;
        }
        auto mixed = make_pair(*hlsl_pair);
        mixed.pixel.language = NativeShaderLanguage::GLSL;
        mixed.pixel.source = glsl_pixel_source;
        mixed.pixel.entry_point = "main";
        auto mixed_result = ext->compile_raster(mixed);
        expect(mixed_result.ok()) << mixed_result.error.c_str();
        auto mixed_shader = ext->create_raster_shader(mixed);
        expect(static_cast<bool>(mixed_shader));
        if (!static_cast<bool>(mixed_shader)) { return; }
        stream << clear_to_black(target).dispatch(width, height)
               << make_draw(mixed_shader, quadrant_meshes(), no_depth_state, nullptr,
                            make_float4(1.f, 1.f, 1.f, 1.f), 1.f)
               << synchronize();
        frame = readback();
        auto mixed_red = probe_pixel(frame, red_x, mid_y);
        expect(close(mixed_red, make_float4(1.f, 0.f, 0.f, 1.f)))
            << ("mixed-language pair must render the red quad, got " +
                to_string(mixed_red))
                   .c_str();
    };

    // ---- 9. two render targets, cleared through the raster extension -----
    "native_raster_multiple_render_targets"_test = [&] {
        // The draw path takes one attachment per render target, and the pixel
        // stage writes one output per attachment. Both targets are cleared with
        // `RasterExt::clear_render_target`, i.e. the command the DSL raster path
        // uses, so this also checks that a native draw interleaves with it.
        auto raster_ext = device.extension<RasterExt>();
        expect(raster_ext != nullptr);
        if (raster_ext == nullptr) { return; }
        auto pair = make_pair(primary->two_target);
        auto shader = ext->create_raster_shader(pair);
        expect(static_cast<bool>(shader));
        // Keep going without crashing when the compile/load of this pair failed:
        // a failed `create_raster_shader` must surface as failed assertions, not
        // as a draw with an invalid handle.
        if (!static_cast<bool>(shader)) { return; }
        auto aux_target = device.create_image<float>(
            PixelStorage::FLOAT4, width, height, 1u, false, true);
        constexpr auto clear_color = make_float4(0.1f, 0.1f, 0.1f, 1.f);
        auto launcher = shader.launcher();
        launcher.add_uniform(make_float4(1.f, 1.f, 1.f, 1.f)).add_uniform(1.f);
        stream << raster_ext->clear_render_target(target.view(), clear_color)
               << raster_ext->clear_render_target(aux_target.view(), clear_color)
               << std::move(launcher).draw(quadrant_meshes(), mesh_format,
                                           Viewport{0u, 0u, width, height},
                                           no_depth_state, nullptr,
                                           target, aux_target)
               << synchronize();
        Frame color_frame;
        Frame aux_frame;
        color_frame.pixels.resize(width * height * 4u);
        aux_frame.pixels.resize(width * height * 4u);
        stream << target.copy_to(luisa::span{color_frame.pixels})
               << aux_target.copy_to(luisa::span{aux_frame.pixels})
               << synchronize();
        // Target 0: the quads' own colours (red left, blue right).
        expect(close(probe_pixel(color_frame, red_x, mid_y), make_float4(1.f, 0.f, 0.f, 1.f)))
            << "target 0 must hold the tinted red quad";
        expect(close(probe_pixel(color_frame, blue_x, mid_y), make_float4(0.f, 0.f, 1.f, 1.f)))
            << "target 0 must hold the blue quad";
        expect(close(probe_pixel(color_frame, middle_x, mid_y), clear_color))
            << "target 0 must keep the clear color in the gap";
        // Target 1: the complement the pixel stage wrote alongside it. A draw
        // that bound only one target would leave this one at the clear color.
        expect(close(probe_pixel(aux_frame, red_x, mid_y), make_float4(0.f, 1.f, 1.f, 1.f)))
            << "target 1 must hold the complement of the red quad";
        expect(close(probe_pixel(aux_frame, blue_x, mid_y), make_float4(1.f, 1.f, 0.f, 1.f)))
            << "target 1 must hold the complement of the blue quad";
        expect(close(probe_pixel(aux_frame, middle_x, mid_y), clear_color))
            << "target 1 must keep the clear color in the gap";
    };

    // ---- 10. fail-closed diagnostics ------------------------------------
    "native_raster_rejects_bad_pairs"_test = [&] {
        // A *pixel* shader whose entry point carries the vertex stage's name: it
        // parses, so the only thing that can reject it is the stage selection
        // (`vs_6_5` refuses `SV_Target`). The pixel stage of the same pair must
        // still compile, which is what makes this a test of the per-stage route
        // rather than of a missing entry point.
        if (hlsl_pair != nullptr) {
            auto wrong_stage = make_pair(
                SourcePair{hlsl_pair->language, hlsl_wrong_stage_source,
                           hlsl_pair->pixel, 0u});
            auto wrong_stage_result = ext->compile_raster(wrong_stage);
            expect(!wrong_stage_result.ok());
            expect(wrong_stage_result.error.find("vertex stage") != luisa::string::npos)
                << wrong_stage_result.error.c_str();
            expect(wrong_stage_result.vertex.binary.empty());
            // The pixel stage of the same pair compiles on its own; only the
            // whole-pair result is cleared on failure (fail closed), so the
            // per-stage error is what shows that the routing - not the source -
            // was wrong.
            expect(wrong_stage_result.pixel.error.empty())
                << wrong_stage_result.pixel.error.c_str();
        }
        // An empty source is rejected without reaching the compiler.
        auto empty_pair = primary->quads;
        auto empty = make_pair(SourcePair{empty_pair.language, "", empty_pair.pixel,
                                          empty_pair.uniform_size});
        auto empty_result = ext->compile_raster(empty);
        expect(!empty_result.ok());
        expect(!empty_result.error.empty());
        // A resource class the route does not bind yet (a sampled image) is
        // rejected instead of producing an unbindable pipeline.
        auto sampler_pair = make_pair(
            SourcePair{primary->quads.language, primary->quads.vertex,
                       primary->sampler_pixel, primary->quads.uniform_size});
        auto sampler_result = ext->compile_raster(sampler_pair);
        expect(!sampler_result.ok());
        expect(sampler_result.vertex.binary.empty() &&
               sampler_result.pixel.binary.empty())
            << "a rejected pair must not report loadable binaries";
        // Two stages that declare the same (space, register) with different
        // resource kinds cannot share one pipeline layout. The message has to name
        // both kinds, so a caller can see *which* declaration has to move; the
        // dialect decides what the second kind is (a GLSL uniform block is a
        // ConstantBuffer, a DirectX `register(u0)` buffer an RWStructuredBuffer).
        auto conflict_result = ext->compile_raster(make_pair(primary->conflict));
        expect(!conflict_result.ok());
        auto conflict_kind =
            primary->conflict.language == NativeShaderLanguage::GLSL ?
                "ConstantBuffer" :
                "RWStructuredBuffer";
        expect(conflict_result.error.find("disagree") != luisa::string::npos &&
               conflict_result.error.find("StructuredBuffer") != luisa::string::npos &&
               conflict_result.error.find(conflict_kind) != luisa::string::npos)
            << conflict_result.error.c_str();
        // A uniform block *smaller* than the module declares cannot be honoured:
        // the shader would read past the range the launcher fills, so this must
        // fail closed instead of being accepted.
        auto too_small = make_pair(primary->quads);
        set_uniform_size(too_small, 8u);
        auto too_small_result = ext->compile_raster(too_small);
        expect(!too_small_result.ok());
        expect(!too_small_result.error.empty());
        expect(too_small_result.vertex.binary.empty() &&
               too_small_result.pixel.binary.empty());
        // The opposite is legal - a pipeline layout may declare a larger range
        // than a stage's block - and must stay accepted, so that an application
        // can report the block size it computes rather than the compiler's.
        auto too_large = make_pair(primary->quads);
        set_uniform_size(too_large, 64u);
        auto too_large_result = ext->compile_raster(too_large);
        expect(too_large_result.ok()) << too_large_result.error.c_str();
        expect(too_large_result.push_constant_size == 64u);
        // DirectX links the two stages by hardware register, so a pixel stage that
        // declares its interpolators in a different order than the vertex stage
        // writes them cannot produce a pipeline; the compile route has to name the
        // varying instead of leaving it to `CreateGraphicsPipelineState`. The other
        // dialect matches the stages by explicit location, so this negative case only
        // exists on dx.
        if (primary == &dx_hlsl_sources) {
            auto mismatched = make_pair(
                SourcePair{primary->quads.language, primary->quads.vertex,
                           hlsl_dx_wrong_order_pixel_source,
                           primary->quads.uniform_size});
            auto mismatched_result = ext->compile_raster(mismatched);
            expect(!mismatched_result.ok());
            expect(mismatched_result.error.find("COLOR") != luisa::string::npos &&
                   mismatched_result.error.find("register") != luisa::string::npos)
                << mismatched_result.error.c_str();
            expect(mismatched_result.vertex.binary.empty() &&
                   mismatched_result.pixel.binary.empty())
                << "a rejected pair must not report loadable binaries";
        }
        // DirectX-only: the array and depth-output cases below are spelled in HLSL,
        // and the rule they check (one register per binding, `SV_Depth` as the
        // pixel-stage witness) is the DirectX side of the route.
        if (primary == &dx_hlsl_sources) {
            // A resource declared as an array spans more than one register, which
            // this route (one root parameter per binding) cannot represent.
            auto array_pair = make_pair(
                SourcePair{primary->quads.language, primary->quads.vertex,
                           hlsl_dx_array_pixel_source,
                           primary->quads.uniform_size});
            auto array_result = ext->compile_raster(array_pair);
            expect(!array_result.ok());
            expect(array_result.error.find("array") != luisa::string::npos)
                << array_result.error.c_str();
            expect(array_result.vertex.binary.empty() &&
                   array_result.pixel.binary.empty())
                << "a rejected pair must not report loadable binaries";
            // A pixel stage that writes only `SV_Depth` serves a depth-only pipeline
            // (no render target), so it must be accepted: the pixel-stage witness is
            // not `SV_Target` alone.
            auto depth_pair = make_pair(
                SourcePair{primary->quads.language, primary->quads.vertex,
                           hlsl_dx_depth_only_pixel_source,
                           primary->quads.uniform_size});
            auto depth_result = ext->compile_raster(depth_pair);
            expect(depth_result.ok()) << depth_result.error.c_str();
            expect(!depth_result.pixel.binary.empty());
        }
        // A usage override must match the merged table (empty for this pair).
        auto valid = ext->compile_raster(make_pair(primary->quads));
        expect(valid.ok()) << valid.error.c_str();
        auto expect_load_rejected = [&](NativeRasterShaderCompileResult const &candidate) {
            auto metadata = ext->load_raster(candidate);
            expect(!metadata.valid());
            // Keep a failed assertion from leaking a handle on a backend that
            // incorrectly accepted the malformed result.
            if (metadata.valid()) { ext->destroy_raster_shader(metadata.handle); }
        };
        // load_raster() is also the boundary for persisted/hand-assembled
        // results. It must not defer stage/language mistakes to PSO creation.
        auto wrong_stage = valid;
        wrong_stage.vertex.stage = NativeShaderStage::Compute;
        expect_load_rejected(wrong_stage);
        auto wrong_language = valid;
        wrong_language.pixel.language = NativeShaderLanguage::CUDA_NVRTC;
        expect_load_rejected(wrong_language);
        auto stage_error = valid;
        stage_error.vertex.error = "injected stage error";
        expect(!stage_error.ok());
        expect_load_rejected(stage_error);
        auto unaligned_uniforms = valid;
        unaligned_uniforms.push_constant_size--;
        expect_load_rejected(unaligned_uniforms);
        auto excessive_uniforms = valid;
        excessive_uniforms.push_constant_size = ~0u - 3u;
        expect_load_rejected(excessive_uniforms);
        auto invalid_object_id = valid;
        invalid_object_id.object_id_bytes = 8u;
        expect_load_rejected(invalid_object_id);
        auto duplicate_binding = valid;
        NativeShaderResourceBinding binding{};
        duplicate_binding.bindings = {binding, binding};
        expect_load_rejected(duplicate_binding);
        if (primary == &glsl_sources) {
            auto malformed_spirv = valid;
            malformed_spirv.vertex.binary.pop_back();
            expect_load_rejected(malformed_spirv);
        }
        std::array<Usage, 1u> bad_override{Usage::READ};
        auto bad = ext->load_raster(valid, luisa::span{bad_override});
        expect(!bad.valid());
        // Destroying an unknown handle is a harmless no-op, not a crash.
        ext->destroy_raster_shader(0x1234u);
    };

    stream << synchronize();
}

}// namespace

int main(int argc, char *argv[]) {
    // The suite can also be run against the validation layer (it wraps the
    // NativeShaderExt forwarders and the draw marking). An environment switch
    // keeps the boost.ut command line untouched:
    //   LUISA_TEST_VALIDATION=1 test_native_shader_raster vk
    auto validation = std::getenv("LUISA_TEST_VALIDATION") != nullptr;
    auto dc = luisa::test::create_device_from_ut(argc, argv, nullptr, validation);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    log_level_verbose();
    test_native_raster(dc->device);
    return 0;
}
