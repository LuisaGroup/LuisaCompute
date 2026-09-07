#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/dsl/struct.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

struct MetalAIRRasterVertex {
    float4 position;
    float4 color;
};

struct MetalAIRRasterVarying {
    float4 position;
    float2 uv;
    float perspective;
    float no_perspective;
    float centroid_perspective;
    float centroid_no_perspective;
    float sample_perspective;
    float sample_no_perspective;
    uint base_instance;

    LUISA_RASTER_VARYING_INTERPOLATION(
        CENTER_PERSPECTIVE,
        CENTER_PERSPECTIVE,
        CENTER_NO_PERSPECTIVE,
        CENTROID_PERSPECTIVE,
        CENTROID_NO_PERSPECTIVE,
        SAMPLE_PERSPECTIVE,
        SAMPLE_NO_PERSPECTIVE,
        FLAT)
};

static_assert(sizeof(MetalAIRRasterVertex) == 32u);
static_assert(offsetof(MetalAIRRasterVertex, position) == 0u);
static_assert(offsetof(MetalAIRRasterVertex, color) == 16u);

LUISA_STRUCT(
    MetalAIRRasterVarying,
    position,
    uv,
    perspective,
    no_perspective,
    centroid_perspective,
    centroid_no_perspective,
    sample_perspective,
    sample_no_perspective,
    base_instance) {};

namespace {

constexpr auto kObjectID = 37u;
constexpr auto kBaseInstance = 7u;
constexpr auto kWrongBaseInstance = 11u;

[[nodiscard]] uint8_t channel(
    const std::array<std::byte, 64u * 64u * 4u> &pixels,
    uint x, uint y, uint c) noexcept {
    return std::to_integer<uint8_t>(pixels[(y * 64u + x) * 4u + c]);
}

[[nodiscard]] std::string read_text_file(
    const std::filesystem::path &path) {
    std::ifstream stream{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{stream},
            std::istreambuf_iterator<char>{}};
}

void expect_contains(const std::string &text, std::string_view token,
                     std::string_view description) {
    expect(text.find(token) != std::string::npos)
        << description << " (missing '" << token << "')";
}

}// namespace

int main(int argc, char *argv[]) {
    // Dump the LLVM module so this integration test can inspect the exact AIR
    // input emitted by the Metal4 backend.
    setenv("LUISA_DUMP_LLVM_IR", "1", 1);
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    constexpr auto width = 64u;
    constexpr auto height = 64u;

    RasterStageKernel vertex = [](
                                   Var<AppData> input,
                                   Float scale,
                                   BufferUInt object_gate) noexcept {
        Var<MetalAIRRasterVarying> output;
        auto object_scale = select(
            0.0f, scale, raster_object_id() == object_gate.read(0u));
        output.position = make_float4(
            input.position.xy() * object_scale,
            0.0f, input.color.w);
        output.uv = input.position.xy();
        auto interpolation_value = input.color.x;
        output.perspective = interpolation_value;
        output.no_perspective = interpolation_value;
        output.centroid_perspective = interpolation_value;
        output.centroid_no_perspective = interpolation_value;
        output.sample_perspective = interpolation_value;
        output.sample_no_perspective = interpolation_value;
        output.base_instance = raster_base_instance();
        return output;
    };
    RasterStageKernel fragment = [](
                                     Var<MetalAIRRasterVarying> input,
                                     Float discard_threshold,
                                     ImageFloat sampled_depth,
                                     ArrayFloat<1024u> large_constants) noexcept {
        auto barycentrics = raster_barycentrics();
        auto derivative = clamp(
            abs(ddx(input.uv.x)) + abs(ddy(input.uv.y)),
            0.0f, 1.0f);
        $if ((raster_object_id() != kObjectID) |
             (input.base_instance != kBaseInstance) |
             (barycentrics.x > discard_threshold)) {
            raster_discard();
        };
        raster_set_z_depth_greater_equal(0.375f);
        auto depth_value = sampled_depth.read(make_uint2(0u)).x;
        return make_float4(
            barycentrics.x, barycentrics.y,
            depth_value + derivative * 0.25f +
                large_constants[0u] * 0.1f,
            select(0.25f, 1.0f, raster_is_front_face()));
    };
    RasterStageKernel fragment_depth_any = [](
                                               Var<MetalAIRRasterVarying> input) noexcept {
        raster_set_z_depth(0.5f);
        return make_float4(input.uv, 0.0f, 1.0f);
    };
    RasterStageKernel fragment_depth_less_equal = [](
                                                      Var<MetalAIRRasterVarying> input) noexcept {
        raster_set_z_depth_less_equal(0.625f);
        return make_float4(input.uv, 0.0f, 1.0f);
    };
    RasterStageKernel fragment_depth_only = [](
                                                Var<MetalAIRRasterVarying>) noexcept {
        raster_set_z_depth(0.4375f);
    };
    RasterStageKernel fragment_interpolation = [](
                                                   Var<MetalAIRRasterVarying> input) noexcept {
        auto centroid_and_sample =
            (input.centroid_perspective +
             input.centroid_no_perspective +
             input.sample_perspective +
             input.sample_no_perspective) *
            0.25f;
        return make_float4(
            input.perspective,
            input.no_perspective,
            centroid_and_sample, 1.0f);
    };
    RasterKernel<decltype(vertex), decltype(fragment)> kernel{vertex, fragment};
    RasterKernel<decltype(vertex), decltype(fragment_depth_any)>
        depth_any_kernel{vertex, fragment_depth_any};
    RasterKernel<decltype(vertex), decltype(fragment_depth_less_equal)>
        depth_less_equal_kernel{vertex, fragment_depth_less_equal};
    RasterKernel<decltype(vertex), decltype(fragment_depth_only)>
        depth_only_kernel{vertex, fragment_depth_only};
    RasterKernel<decltype(vertex), decltype(fragment_interpolation)>
        interpolation_kernel{vertex, fragment_interpolation};

    MeshFormat mesh_format;
    const VertexAttribute attributes[]{
        {VertexAttributeType::Position, PixelFormat::RGBA32F},
        {VertexAttributeType::Color, PixelFormat::RGBA32F},
    };
    mesh_format.emplace_vertex_stream(attributes);

    auto nonce = std::chrono::steady_clock::now()
                     .time_since_epoch()
                     .count();
    auto dump_prefix = std::filesystem::temp_directory_path() /
                       ("luisa_metal_raster_air_" + std::to_string(nonce));
    auto dump_prefix_string = dump_prefix.string();
    ShaderOption option{};
    option.enable_cache = false;
    option.name = luisa::string{
        dump_prefix_string.data(), dump_prefix_string.size()};
    auto shader = dc->device.compile(kernel, mesh_format, option);
    auto depth_only_prefix = dump_prefix_string + ".depth_only";
    ShaderOption depth_only_option{};
    depth_only_option.enable_cache = false;
    depth_only_option.name = luisa::string{
        depth_only_prefix.data(), depth_only_prefix.size()};
    auto depth_only_shader = dc->device.compile(
        depth_only_kernel, mesh_format, depth_only_option);
    auto interpolation_prefix = dump_prefix_string + ".interpolation";
    ShaderOption interpolation_option{};
    interpolation_option.enable_cache = false;
    interpolation_option.name = luisa::string{
        interpolation_prefix.data(), interpolation_prefix.size()};
    auto interpolation_shader = dc->device.compile(
        interpolation_kernel, mesh_format, interpolation_option);

    auto check_depth_qualifier = [&]<typename Kernel>(
                                     const Kernel &qualifier_kernel,
                                     std::string_view suffix,
                                     std::string_view qualifier) {
        auto qualifier_prefix = dump_prefix_string + std::string{suffix};
        ShaderOption qualifier_option{};
        qualifier_option.enable_cache = false;
        qualifier_option.name = luisa::string{
            qualifier_prefix.data(), qualifier_prefix.size()};
        static_cast<void>(dc->device.compile(
            qualifier_kernel, mesh_format, qualifier_option));
        auto qualifier_ir_path = std::filesystem::path{
            qualifier_prefix + ".fragment.air.ll"};
        expect(std::filesystem::is_regular_file(qualifier_ir_path))
            << "fragment depth AIR LLVM dump was not written";
        auto qualifier_ir = read_text_file(qualifier_ir_path);
        expect_contains(
            qualifier_ir, qualifier,
            "fragment depth qualifier metadata must match the DSL operation");
        std::error_code ignored;
        std::filesystem::remove(qualifier_ir_path, ignored);
        std::filesystem::remove(
            qualifier_prefix + ".vertex.air.ll", ignored);
    };
    check_depth_qualifier(
        depth_any_kernel, ".depth_any", "!\"air.any\"");
    check_depth_qualifier(
        depth_less_equal_kernel, ".depth_less_equal", "!\"air.less\"");

    auto vertex_ir_path = std::filesystem::path{
        dump_prefix_string + ".vertex.air.ll"};
    auto fragment_ir_path = std::filesystem::path{
        dump_prefix_string + ".fragment.air.ll"};
    auto depth_only_fragment_ir_path = std::filesystem::path{
        depth_only_prefix + ".fragment.air.ll"};
    auto interpolation_fragment_ir_path = std::filesystem::path{
        interpolation_prefix + ".fragment.air.ll"};
    expect(std::filesystem::is_regular_file(vertex_ir_path))
        << "vertex AIR LLVM dump was not written";
    expect(std::filesystem::is_regular_file(fragment_ir_path))
        << "fragment AIR LLVM dump was not written";
    expect(std::filesystem::is_regular_file(depth_only_fragment_ir_path))
        << "depth-only fragment AIR LLVM dump was not written";
    expect(std::filesystem::is_regular_file(
        interpolation_fragment_ir_path))
        << "interpolation fragment AIR LLVM dump was not written";
    auto vertex_ir = read_text_file(vertex_ir_path);
    auto fragment_ir = read_text_file(fragment_ir_path);
    auto depth_only_fragment_ir = read_text_file(
        depth_only_fragment_ir_path);
    auto interpolation_fragment_ir = read_text_file(
        interpolation_fragment_ir_path);

    expect_contains(vertex_ir, "@vertex_main(",
                    "vertex entry point must be emitted");
    expect_contains(vertex_ir, "!air.vertex = !{",
                    "vertex AIR named metadata must be emitted");
    expect_contains(vertex_ir, "!\"air.vertex_input\"",
                    "vertex attributes must use AIR vertex-input metadata");
    expect_contains(vertex_ir, "!\"air.location_index\"",
                    "vertex attributes must carry location indices");
    expect_contains(vertex_ir, "!\"air.vertex_id\"",
                    "vertex ID builtin metadata must be emitted");
    expect_contains(vertex_ir, "!\"air.instance_id\"",
                    "instance ID builtin metadata must be emitted");
    expect_contains(vertex_ir, "!\"air.base_instance\"",
                    "base-instance builtin metadata must be emitted");
    expect_contains(vertex_ir, "i32 noundef %base_instance",
                    "base instance must use AIR's physical uint ABI");
    expect_contains(vertex_ir, "!\"air.indirect_buffer\"",
                    "vertex root arguments must use the indirect-buffer ABI");
    expect_contains(vertex_ir, "%object_id",
                    "vertex entry must receive the per-draw object ID buffer");
    expect_contains(vertex_ir, "!\"air.arg_name\", !\"object_id\"",
                    "vertex object ID buffer metadata must be emitted");

    expect_contains(fragment_ir, "@fragment_main(",
                    "fragment entry point must be emitted");
    expect_contains(fragment_ir, "!air.fragment = !{",
                    "fragment AIR named metadata must be emitted");
    expect_contains(fragment_ir, "!\"air.position\"",
                    "fragment position input metadata must be emitted");
    expect_contains(fragment_ir, "!\"air.fragment_input\"",
                    "fragment varyings must use AIR fragment-input metadata");
    expect_contains(fragment_ir, "!\"air.flat\"",
                    "integer base-instance varying must use flat interpolation");
    expect_contains(fragment_ir, "!\"air.primitive_id\"",
                    "fragment primitive ID builtin metadata must be emitted");
    expect_contains(fragment_ir, "!\"air.barycentric_coord\"",
                    "fragment barycentric builtin metadata must be emitted");
    expect_contains(fragment_ir, "!\"air.front_facing\"",
                    "fragment front-facing builtin metadata must be emitted");
    expect_contains(fragment_ir, "i1 noundef %front_facing",
                    "fragment front-facing input must use AIR's physical i1 ABI");
    expect_contains(fragment_ir, "!\"air.render_target\"",
                    "fragment color output metadata must be emitted");
    expect_contains(fragment_ir, "!\"air.depth\"",
                    "fragment depth output metadata must be emitted");
    expect_contains(fragment_ir, "!\"air.depth_qualifier\"",
                    "fragment depth qualifier metadata must be emitted");
    expect_contains(fragment_ir, "!\"air.greater\"",
                    "greater-equal shader depth must use AIR's greater qualifier");
    expect_contains(fragment_ir, "!\"air.indirect_buffer\"",
                    "fragment root arguments must use the indirect-buffer ABI");
    expect_contains(fragment_ir, "%object_id",
                    "fragment entry must receive the per-draw object ID buffer");
    expect_contains(fragment_ir, "!\"air.arg_name\", !\"object_id\"",
                    "fragment object ID buffer metadata must be emitted");
    expect_contains(fragment_ir, "@air.dfdx.f32",
                    "fragment ddx must lower to the AIR derivative intrinsic");
    expect_contains(fragment_ir, "@air.dfdy.f32",
                    "fragment ddy must lower to the AIR derivative intrinsic");
    expect_contains(fragment_ir, "@air.discard_fragment",
                    "fragment discard must lower to the AIR discard intrinsic");
    expect_contains(depth_only_fragment_ir,
                    "define <{ float }> @fragment_main(",
                    "depth-only fragment must use Apple's packed depth return ABI");
    expect_contains(depth_only_fragment_ir, "!\"air.depth\"",
                    "depth-only fragment metadata must emit a depth output");
    expect(depth_only_fragment_ir.find("!\"air.render_target\"") ==
           std::string::npos)
        << "depth-only fragment unexpectedly emitted a color target";
    expect_contains(
        interpolation_fragment_ir,
        "!\"air.center\", !\"air.perspective\"",
        "center-perspective varying metadata must be emitted");
    expect_contains(
        interpolation_fragment_ir,
        "!\"air.center\", !\"air.no_perspective\"",
        "center-no-perspective varying metadata must be emitted");
    expect_contains(
        interpolation_fragment_ir,
        "!\"air.centroid\", !\"air.perspective\"",
        "centroid-perspective varying metadata must be emitted");
    expect_contains(
        interpolation_fragment_ir,
        "!\"air.centroid\", !\"air.no_perspective\"",
        "centroid-no-perspective varying metadata must be emitted");
    expect_contains(
        interpolation_fragment_ir,
        "!\"air.sample\", !\"air.perspective\"",
        "sample-perspective varying metadata must be emitted");
    expect_contains(
        interpolation_fragment_ir,
        "!\"air.sample\", !\"air.no_perspective\"",
        "sample-no-perspective varying metadata must be emitted");

    const std::array vertices{
        MetalAIRRasterVertex{
            .position = {-0.8f, -0.8f, 0.0f, 1.0f},
            .color = {1.0f, 0.0f, 0.0f, 1.0f}},
        MetalAIRRasterVertex{
            .position = {0.8f, -0.8f, 0.0f, 1.0f},
            .color = {0.0f, 1.0f, 0.0f, 1.0f}},
        MetalAIRRasterVertex{
            .position = {0.0f, 0.8f, 0.0f, 1.0f},
            .color = {0.0f, 0.0f, 1.0f, 1.0f}},
    };
    const std::array indices{0u, 1u, 2u};
    const std::array interpolation_vertices{
        MetalAIRRasterVertex{
            .position = {-0.8f, -0.8f, 0.0f, 1.0f},
            .color = {0.0f, 0.0f, 0.0f, 1.0f}},
        MetalAIRRasterVertex{
            .position = {0.8f, -0.8f, 0.0f, 1.0f},
            .color = {0.0f, 0.0f, 0.0f, 1.0f}},
        MetalAIRRasterVertex{
            .position = {0.0f, 3.2f, 0.0f, 4.0f},
            .color = {1.0f, 0.0f, 0.0f, 4.0f}},
    };
    auto vertex_buffer = dc->device.create_buffer<MetalAIRRasterVertex>(
        vertices.size());
    auto interpolation_vertex_buffer =
        dc->device.create_buffer<MetalAIRRasterVertex>(
            interpolation_vertices.size());
    auto index_buffer = dc->device.create_buffer<uint>(indices.size());
    auto object_gate = dc->device.create_buffer<uint>(1u);
    auto sampled_depth = dc->device.create_depth_buffer(
        DepthFormat::D32, make_uint2(1u));
    auto output_depth = dc->device.create_depth_buffer(
        DepthFormat::D32, make_uint2(width, height));
    auto depth_sample_buffer = dc->device.create_buffer<float>(1u);
    Kernel1D read_depth = [](ImageFloat depth, BufferFloat output) noexcept {
        output.write(0u, depth.read(make_uint2(width / 2u, height / 2u)).x);
    };
    auto read_depth_shader = dc->device.compile(read_depth);
    auto render_target = dc->device.create_image<float>(
        PixelStorage::BYTE4, width, height, 1u, false, true);
    auto stream = dc->device.create_stream(StreamTag::GRAPHICS);
    auto raster = dc->device.extension<RasterExt>();

    VertexBufferView vertex_view{vertex_buffer};
    VertexBufferView interpolation_vertex_view{
        interpolation_vertex_buffer};
    luisa::vector<RasterMesh> meshes;
    meshes.emplace_back(
        luisa::span<const VertexBufferView>{&vertex_view, 1u},
        index_buffer.view(), 1u, kObjectID, 0, kBaseInstance);
    RasterState state{};
    state.cull_mode = CullMode::None;
    state.depth_state = DepthState{
        .enable_depth = true,
        .comparison = Comparison::Always,
        .write = true};

    std::array<std::byte, width * height * 4u> pixels{};
    std::array<float, 1u> depth_sample{};
    const std::array gate_value{kObjectID};
    std::array<float, 1024u> large_constants{};
    large_constants[0u] = 0.5f;
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << interpolation_vertex_buffer.copy_from(
                  luisa::span{interpolation_vertices})
           << index_buffer.copy_from(luisa::span{indices})
           << object_gate.copy_from(luisa::span{gate_value})
           << sampled_depth.clear(0.25f)
           << output_depth.clear(1.0f)
           << raster->clear_render_target(
                  render_target.view(), make_float4(0.0f, 0.0f, 0.0f, 1.0f))
           << shader(1.0f, object_gate, 0.72f,
                     sampled_depth.to_img(), large_constants)
                  .draw(
                      std::move(meshes), mesh_format,
                      Viewport{0u, 0u, width, height}, state,
                      &output_depth, render_target)
           << render_target.copy_to(luisa::span{pixels})
           << read_depth_shader(output_depth.to_img(), depth_sample_buffer)
                  .dispatch(1u)
           << depth_sample_buffer.copy_to(luisa::span{depth_sample})
           << synchronize();

    expect(depth_sample[0u] > 0.3749f && depth_sample[0u] < 0.3751f)
        << "shader-written depth=" << depth_sample[0u];

    // A fragment stage with no color return must still form a valid AIR
    // entry when it writes shader depth. Bind only D32 here: any hidden color
    // attachment requirement or a lost packed-depth return becomes visible.
    std::array<float, 1u> depth_only_jit_sample{};
    luisa::vector<RasterMesh> depth_only_jit_meshes;
    depth_only_jit_meshes.emplace_back(
        luisa::span<const VertexBufferView>{&vertex_view, 1u},
        index_buffer.view(), 1u, kObjectID, 0, kBaseInstance);
    stream << output_depth.clear(1.0f)
           << depth_only_shader(1.0f, object_gate)
                  .draw(
                      std::move(depth_only_jit_meshes), mesh_format,
                      Viewport{0u, 0u, width, height}, state,
                      &output_depth)
           << read_depth_shader(output_depth.to_img(), depth_sample_buffer)
                  .dispatch(1u)
           << depth_sample_buffer.copy_to(
                  luisa::span{depth_only_jit_sample})
           << synchronize();
    expect(depth_only_jit_sample[0u] > 0.4374f &&
           depth_only_jit_sample[0u] < 0.4376f)
        << "depth-only JIT value=" << depth_only_jit_sample[0u];

    // Give the top vertex a larger clip-space W while preserving its NDC
    // position. At the image center, perspective correction pulls the red
    // varying toward the two W=1 bottom vertices, while the green
    // no-perspective varying remains screen-linear. Centroid/sample fields
    // are consumed in blue so their AIR inputs survive through GPU execution.
    std::array<std::byte, width * height * 4u>
        interpolation_jit_pixels{};
    luisa::vector<RasterMesh> interpolation_jit_meshes;
    interpolation_jit_meshes.emplace_back(
        luisa::span<const VertexBufferView>{
            &interpolation_vertex_view, 1u},
        index_buffer.view(), 1u, kObjectID, 0, kBaseInstance);
    stream << output_depth.clear(1.0f)
           << raster->clear_render_target(
                  render_target.view(),
                  make_float4(0.0f, 0.0f, 0.0f, 1.0f))
           << interpolation_shader(1.0f, object_gate)
                  .draw(
                      std::move(interpolation_jit_meshes), mesh_format,
                      Viewport{0u, 0u, width, height}, state,
                      &output_depth, render_target)
           << render_target.copy_to(
                  luisa::span{interpolation_jit_pixels})
           << synchronize();
    auto interpolation_red = channel(
        interpolation_jit_pixels, width / 2u, height / 2u, 0u);
    auto interpolation_green = channel(
        interpolation_jit_pixels, width / 2u, height / 2u, 1u);
    auto interpolation_blue = channel(
        interpolation_jit_pixels, width / 2u, height / 2u, 2u);
    expect(interpolation_red > 35u && interpolation_red < 75u)
        << "perspective center value="
        << static_cast<uint>(interpolation_red);
    expect(interpolation_green > 110u && interpolation_green < 150u)
        << "no-perspective center value="
        << static_cast<uint>(interpolation_green);
    expect(interpolation_green > interpolation_red + 45u)
        << "perspective and no-perspective varyings did not diverge";
    expect(interpolation_blue > 35u && interpolation_blue < 150u)
        << "centroid/sample center value="
        << static_cast<uint>(interpolation_blue);

    auto colored_pixels = 0u;
    auto max_red = uint8_t{0u};
    auto max_green = uint8_t{0u};
    auto max_blue = uint8_t{0u};
    for (auto y = 0u; y < height; y++) {
        for (auto x = 0u; x < width; x++) {
            auto red = channel(pixels, x, y, 0u);
            auto green = channel(pixels, x, y, 1u);
            auto blue = channel(pixels, x, y, 2u);
            if (red > 2u || green > 2u || blue > 2u) {
                colored_pixels++;
                max_red = std::max(max_red, red);
                max_green = std::max(max_green, green);
                max_blue = std::max(max_blue, blue);
            }
        }
    }

    // A visible triangle proves that the vertex-stage object ID and dynamic
    // root constant reached AIR; either failure collapses every vertex.
    expect(colored_pixels > 700u);
    expect(colored_pixels < 1500u);
    // The fragment discards barycentric-x values above 0.72. The surviving
    // red channel therefore remains capped below the undiscarded vertex peak.
    expect(max_red > 120u) << "max_red=" << static_cast<uint>(max_red);
    expect(max_red < 210u) << "max_red=" << static_cast<uint>(max_red);
    expect(max_green > 180u) << "max_green=" << static_cast<uint>(max_green);
    // The blue channel combines a dynamic dfdx/dfdy result, a D32 depth
    // texture read through DepthBuffer::to_img(), and a 4 KiB uniform array.
    // The complete root block exceeds Metal's inline-byte limit, so this also
    // exercises the upload-buffer root path and the shared depth/image ABI.
    expect(max_blue > 50u) << "max_blue=" << static_cast<uint>(max_blue);
    expect(max_blue < 100u) << "max_blue=" << static_cast<uint>(max_blue);
    // Center is covered; an outer corner remains at the clear color.
    expect(channel(pixels, width / 2u, height / 2u, 0u) +
               channel(pixels, width / 2u, height / 2u, 1u) +
               channel(pixels, width / 2u, height / 2u, 2u) >
           20u);
    expect(channel(pixels, 0u, 0u, 0u) == 0u);
    expect(channel(pixels, 0u, 0u, 1u) == 0u);
    expect(channel(pixels, 0u, 0u, 2u) == 0u);
    expect(channel(pixels, 0u, 0u, 3u) == 255u);

    // Exercise the complete raster AOT boundary with the same large root
    // block and depth-image resource. The loaded archive must reproduce the
    // JIT image exactly.
    auto jit_pixels = pixels;
    auto jit_front_alpha = channel(
        jit_pixels, width / 2u, height / 2u, 3u);
    expect(jit_front_alpha == 64u || jit_front_alpha == 255u)
        << "front-facing alpha=" << static_cast<uint>(jit_front_alpha);

    // Flipping the declared front winding must invert [[front_facing]] while
    // culling stays disabled. This catches a constant-folded or incorrectly
    // wired AIR argument even when the triangle still renders visibly.
    auto opposite_state = state;
    opposite_state.front_counter_clockwise =
        !state.front_counter_clockwise;
    std::array<std::byte, width * height * 4u> opposite_jit_pixels{};
    luisa::vector<RasterMesh> opposite_jit_meshes;
    opposite_jit_meshes.emplace_back(
        luisa::span<const VertexBufferView>{&vertex_view, 1u},
        static_cast<uint>(vertices.size()), 1u, kObjectID, 0,
        kBaseInstance);
    stream << output_depth.clear(1.0f)
           << raster->clear_render_target(
                  render_target.view(),
                  make_float4(0.0f, 0.0f, 0.0f, 1.0f))
           << shader(1.0f, object_gate, 0.72f,
                     sampled_depth.to_img(), large_constants)
                  .draw(
                      std::move(opposite_jit_meshes), mesh_format,
                      Viewport{0u, 0u, width, height}, opposite_state,
                      &output_depth, render_target)
           << render_target.copy_to(
                  luisa::span{opposite_jit_pixels})
           << synchronize();
    auto opposite_front_alpha = channel(
        opposite_jit_pixels, width / 2u, height / 2u, 3u);
    expect(opposite_front_alpha == 64u ||
           opposite_front_alpha == 255u)
        << "opposite-winding alpha="
        << static_cast<uint>(opposite_front_alpha);
    expect(opposite_front_alpha != jit_front_alpha)
        << "front-facing did not change when winding was inverted";

    // A different draw-time base instance must reach the vertex AIR builtin.
    // The vertex stage passes it as a flat varying and the fragment rejects
    // the mismatch, so a hard-coded or ignored base-instance value is visible.
    std::array<std::byte, width * height * 4u> wrong_base_jit_pixels{};
    luisa::vector<RasterMesh> wrong_base_jit_meshes;
    wrong_base_jit_meshes.emplace_back(
        luisa::span<const VertexBufferView>{&vertex_view, 1u},
        index_buffer.view(), 1u, kObjectID, 0, kWrongBaseInstance);
    stream << output_depth.clear(1.0f)
           << raster->clear_render_target(
                  render_target.view(),
                  make_float4(0.0f, 0.0f, 0.0f, 1.0f))
           << shader(1.0f, object_gate, 0.72f,
                     sampled_depth.to_img(), large_constants)
                  .draw(
                      std::move(wrong_base_jit_meshes), mesh_format,
                      Viewport{0u, 0u, width, height}, state,
                      &output_depth, render_target)
           << render_target.copy_to(
                  luisa::span{wrong_base_jit_pixels})
           << synchronize();
    auto wrong_base_colored_pixels = 0u;
    for (auto y = 0u; y < height; y++) {
        for (auto x = 0u; x < width; x++) {
            wrong_base_colored_pixels +=
                channel(wrong_base_jit_pixels, x, y, 0u) != 0u ||
                channel(wrong_base_jit_pixels, x, y, 1u) != 0u ||
                channel(wrong_base_jit_pixels, x, y, 2u) != 0u;
        }
    }
    expect(wrong_base_colored_pixels == 0u)
        << "wrong-base-instance draw produced "
        << wrong_base_colored_pixels << " colored pixel(s)";

    auto archive_path = std::filesystem::path{
        dump_prefix_string + ".raster.air.archive"};
    auto archive_path_string = archive_path.string();
    dc->device.compile_to(
        kernel, mesh_format, archive_path_string);
    expect(std::filesystem::is_regular_file(archive_path))
        << "raster AIR archive was not written";
    auto aot_shader = dc->device.load_raster_shader<
        float, Buffer<uint>, float, Image<float>,
        std::array<float, 1024u>>(archive_path_string);
    expect(static_cast<bool>(aot_shader))
        << "raster AIR archive failed to load";
    if (aot_shader) {
        pixels.fill(std::byte{});
        luisa::vector<RasterMesh> aot_meshes;
        aot_meshes.emplace_back(
            luisa::span<const VertexBufferView>{&vertex_view, 1u},
            index_buffer.view(), 1u, kObjectID, 0, kBaseInstance);
        stream << sampled_depth.clear(0.25f)
               << output_depth.clear(1.0f)
               << raster->clear_render_target(
                      render_target.view(),
                      make_float4(0.0f, 0.0f, 0.0f, 1.0f))
               << aot_shader(1.0f, object_gate, 0.72f,
                             sampled_depth.to_img(), large_constants)
                      .draw(
                          std::move(aot_meshes), mesh_format,
                          Viewport{0u, 0u, width, height}, state,
                          &output_depth, render_target)
               << render_target.copy_to(luisa::span{pixels})
               << read_depth_shader(output_depth.to_img(), depth_sample_buffer)
                      .dispatch(1u)
               << depth_sample_buffer.copy_to(luisa::span{depth_sample})
               << synchronize();
        expect(static_cast<bool>(pixels == jit_pixels))
            << "AOT raster output differs from JIT output";
        expect(depth_sample[0u] > 0.3749f && depth_sample[0u] < 0.3751f)
            << "AOT shader-written depth=" << depth_sample[0u];

        std::array<std::byte, width * height * 4u>
            opposite_aot_pixels{};
        luisa::vector<RasterMesh> opposite_aot_meshes;
        opposite_aot_meshes.emplace_back(
            luisa::span<const VertexBufferView>{&vertex_view, 1u},
            static_cast<uint>(vertices.size()), 1u, kObjectID, 0,
            kBaseInstance);
        stream << output_depth.clear(1.0f)
               << raster->clear_render_target(
                      render_target.view(),
                      make_float4(0.0f, 0.0f, 0.0f, 1.0f))
               << aot_shader(1.0f, object_gate, 0.72f,
                             sampled_depth.to_img(), large_constants)
                      .draw(
                          std::move(opposite_aot_meshes), mesh_format,
                          Viewport{0u, 0u, width, height},
                          opposite_state, &output_depth, render_target)
               << render_target.copy_to(
                      luisa::span{opposite_aot_pixels})
               << synchronize();
        expect(static_cast<bool>(
            opposite_aot_pixels == opposite_jit_pixels))
            << "AOT opposite-winding output differs from JIT output";

        std::array<std::byte, width * height * 4u>
            wrong_base_aot_pixels{};
        luisa::vector<RasterMesh> wrong_base_aot_meshes;
        wrong_base_aot_meshes.emplace_back(
            luisa::span<const VertexBufferView>{&vertex_view, 1u},
            index_buffer.view(), 1u, kObjectID, 0, kWrongBaseInstance);
        stream << output_depth.clear(1.0f)
               << raster->clear_render_target(
                      render_target.view(),
                      make_float4(0.0f, 0.0f, 0.0f, 1.0f))
               << aot_shader(1.0f, object_gate, 0.72f,
                             sampled_depth.to_img(), large_constants)
                      .draw(
                          std::move(wrong_base_aot_meshes), mesh_format,
                          Viewport{0u, 0u, width, height}, state,
                          &output_depth, render_target)
               << render_target.copy_to(
                      luisa::span{wrong_base_aot_pixels})
               << synchronize();
        expect(static_cast<bool>(
            wrong_base_aot_pixels == wrong_base_jit_pixels))
            << "AOT wrong-base-instance output differs from JIT output";
    }

    auto interpolation_archive_path = std::filesystem::path{
        dump_prefix_string + ".interpolation.raster.air.archive"};
    auto interpolation_archive_path_string =
        interpolation_archive_path.string();
    dc->device.compile_to(
        interpolation_kernel, mesh_format,
        interpolation_archive_path_string);
    expect(std::filesystem::is_regular_file(
        interpolation_archive_path))
        << "interpolation raster AIR archive was not written";
    auto interpolation_aot_shader =
        dc->device.load_raster_shader<float, Buffer<uint>>(
            interpolation_archive_path_string);
    expect(static_cast<bool>(interpolation_aot_shader))
        << "interpolation raster AIR archive failed to load";
    if (interpolation_aot_shader) {
        std::array<std::byte, width * height * 4u>
            interpolation_aot_pixels{};
        luisa::vector<RasterMesh> interpolation_aot_meshes;
        interpolation_aot_meshes.emplace_back(
            luisa::span<const VertexBufferView>{
                &interpolation_vertex_view, 1u},
            index_buffer.view(), 1u, kObjectID, 0, kBaseInstance);
        stream << output_depth.clear(1.0f)
               << raster->clear_render_target(
                      render_target.view(),
                      make_float4(0.0f, 0.0f, 0.0f, 1.0f))
               << interpolation_aot_shader(1.0f, object_gate)
                      .draw(
                          std::move(interpolation_aot_meshes),
                          mesh_format,
                          Viewport{0u, 0u, width, height}, state,
                          &output_depth, render_target)
               << render_target.copy_to(
                      luisa::span{interpolation_aot_pixels})
               << synchronize();
        expect(static_cast<bool>(
            interpolation_aot_pixels ==
            interpolation_jit_pixels))
            << "AOT interpolation output differs from JIT output";
    }

    auto depth_only_archive_path = std::filesystem::path{
        dump_prefix_string + ".depth_only.raster.air.archive"};
    auto depth_only_archive_path_string =
        depth_only_archive_path.string();
    dc->device.compile_to(
        depth_only_kernel, mesh_format,
        depth_only_archive_path_string);
    expect(std::filesystem::is_regular_file(depth_only_archive_path))
        << "depth-only raster AIR archive was not written";
    auto depth_only_aot_shader =
        dc->device.load_raster_shader<float, Buffer<uint>>(
            depth_only_archive_path_string);
    expect(static_cast<bool>(depth_only_aot_shader))
        << "depth-only raster AIR archive failed to load";
    if (depth_only_aot_shader) {
        std::array<float, 1u> depth_only_aot_sample{};
        luisa::vector<RasterMesh> depth_only_aot_meshes;
        depth_only_aot_meshes.emplace_back(
            luisa::span<const VertexBufferView>{&vertex_view, 1u},
            index_buffer.view(), 1u, kObjectID, 0, kBaseInstance);
        stream << output_depth.clear(1.0f)
               << depth_only_aot_shader(1.0f, object_gate)
                      .draw(
                          std::move(depth_only_aot_meshes), mesh_format,
                          Viewport{0u, 0u, width, height}, state,
                          &output_depth)
               << read_depth_shader(
                      output_depth.to_img(), depth_sample_buffer)
                      .dispatch(1u)
               << depth_sample_buffer.copy_to(
                      luisa::span{depth_only_aot_sample})
               << synchronize();
        expect(depth_only_aot_sample[0u] > 0.4374f &&
               depth_only_aot_sample[0u] < 0.4376f)
            << "depth-only AOT value=" << depth_only_aot_sample[0u];
        expect(depth_only_aot_sample[0u] ==
               depth_only_jit_sample[0u])
            << "depth-only AOT output differs from JIT output";
    }
    std::error_code ignored;
    std::filesystem::remove(vertex_ir_path, ignored);
    std::filesystem::remove(fragment_ir_path, ignored);
    std::filesystem::remove(depth_only_fragment_ir_path, ignored);
    std::filesystem::remove(
        interpolation_fragment_ir_path, ignored);
    std::filesystem::remove(
        depth_only_prefix + ".vertex.air.ll", ignored);
    std::filesystem::remove(
        interpolation_prefix + ".vertex.air.ll", ignored);
    std::filesystem::remove(archive_path, ignored);
    std::filesystem::remove(
        archive_path_string + ".vertex.air.ll", ignored);
    std::filesystem::remove(
        archive_path_string + ".fragment.air.ll", ignored);
    std::filesystem::remove(interpolation_archive_path, ignored);
    std::filesystem::remove(
        interpolation_archive_path_string + ".vertex.air.ll", ignored);
    std::filesystem::remove(
        interpolation_archive_path_string + ".fragment.air.ll", ignored);
    std::filesystem::remove(depth_only_archive_path, ignored);
    std::filesystem::remove(
        depth_only_archive_path_string + ".vertex.air.ll", ignored);
    std::filesystem::remove(
        depth_only_archive_path_string + ".fragment.air.ll", ignored);
    return 0;
}
