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
};

static_assert(sizeof(MetalAIRRasterVertex) == 32u);
static_assert(offsetof(MetalAIRRasterVertex, position) == 0u);
static_assert(offsetof(MetalAIRRasterVertex, color) == 16u);

LUISA_STRUCT(MetalAIRRasterVarying, position, uv) {};

namespace {

constexpr auto kObjectID = 37u;

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
            input.position.xy() * object_scale, 0.0f, 1.0f);
        output.uv = input.position.xy();
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
             (barycentrics.x > discard_threshold)) {
            raster_discard();
        };
        raster_set_z_depth_greater_equal(0.375f);
        auto depth_value = sampled_depth.read(make_uint2(0u)).x;
        return make_float4(
            barycentrics.x, barycentrics.y,
            depth_value + derivative * 0.25f +
                large_constants[0u] * 0.1f,
            1.0f);
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
    RasterKernel<decltype(vertex), decltype(fragment)> kernel{vertex, fragment};
    RasterKernel<decltype(vertex), decltype(fragment_depth_any)>
        depth_any_kernel{vertex, fragment_depth_any};
    RasterKernel<decltype(vertex), decltype(fragment_depth_less_equal)>
        depth_less_equal_kernel{vertex, fragment_depth_less_equal};

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
    expect(std::filesystem::is_regular_file(vertex_ir_path))
        << "vertex AIR LLVM dump was not written";
    expect(std::filesystem::is_regular_file(fragment_ir_path))
        << "fragment AIR LLVM dump was not written";
    auto vertex_ir = read_text_file(vertex_ir_path);
    auto fragment_ir = read_text_file(fragment_ir_path);

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
    expect_contains(fragment_ir, "!\"air.primitive_id\"",
                    "fragment primitive ID builtin metadata must be emitted");
    expect_contains(fragment_ir, "!\"air.barycentric_coord\"",
                    "fragment barycentric builtin metadata must be emitted");
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
    auto vertex_buffer = dc->device.create_buffer<MetalAIRRasterVertex>(
        vertices.size());
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
    luisa::vector<RasterMesh> meshes;
    meshes.emplace_back(
        luisa::span<const VertexBufferView>{&vertex_view, 1u},
        static_cast<uint>(vertices.size()), 1u, kObjectID);
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
            static_cast<uint>(vertices.size()), 1u, kObjectID);
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
    }
    std::error_code ignored;
    std::filesystem::remove(vertex_ir_path, ignored);
    std::filesystem::remove(fragment_ir_path, ignored);
    std::filesystem::remove(archive_path, ignored);
    std::filesystem::remove(
        archive_path_string + ".vertex.air.ll", ignored);
    std::filesystem::remove(
        archive_path_string + ".fragment.air.ll", ignored);
    return 0;
}
