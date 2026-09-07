// Metal4 raster stencil integration coverage.
// This test covers D24S8 capability fallback, D32S8A24, stencil attachment
// load/store/clear, comparison, and depth-failure state through real draws.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/dsl/struct.h>
#include <luisa/luisa-compute.h>

#include <array>
#include <cstddef>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct Metal4StencilVarying {
    float4 position;
};

LUISA_STRUCT(Metal4StencilVarying, position) {};

namespace {

constexpr auto kWidth = 32u;
constexpr auto kHeight = 32u;

[[nodiscard]] StencilFaceOp stencil_face(
    Comparison comparison,
    StencilOp stencil_fail = StencilOp::Keep,
    StencilOp depth_fail = StencilOp::Keep,
    StencilOp pass = StencilOp::Keep) noexcept {
    return StencilFaceOp{
        .stencil_fail_op = stencil_fail,
        .depth_fail_op = depth_fail,
        .pass_op = pass,
        .comparison = comparison};
}

[[nodiscard]] RasterState raster_state(
    uint8_t reference,
    Comparison stencil_comparison,
    Comparison depth_comparison,
    StencilOp stencil_fail = StencilOp::Keep,
    StencilOp depth_fail = StencilOp::Keep,
    StencilOp pass = StencilOp::Keep,
    uint8_t read_mask = 0xffu,
    uint8_t write_mask = 0xffu) noexcept {
    auto face = stencil_face(
        stencil_comparison, stencil_fail, depth_fail, pass);
    RasterState state{};
    state.cull_mode = CullMode::None;
    state.depth_state = DepthState{
        .enable_depth = true,
        .comparison = depth_comparison,
        .write = true};
    state.stencil_state = StencilState{
        .enable_stencil = true,
        .front_face_op = face,
        .back_face_op = face,
        .read_mask = read_mask,
        .write_mask = write_mask,
        .reference = reference};
    return state;
}

[[nodiscard]] uint32_t colored_pixel_count(
    const std::array<std::byte, kWidth * kHeight * 4u> &pixels) noexcept {
    auto count = 0u;
    for (auto i = 0u; i < kWidth * kHeight; i++) {
        auto offset = i * 4u;
        count += std::to_integer<uint8_t>(pixels[offset]) > 200u &&
                 std::to_integer<uint8_t>(pixels[offset + 1u]) < 8u &&
                 std::to_integer<uint8_t>(pixels[offset + 2u]) < 8u;
    }
    return count;
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    RasterStageKernel vertex = [](
                                   Var<AppData> input) noexcept {
        Var<Metal4StencilVarying> output;
        output.position = make_float4(input.position, 1.0f);
        return output;
    };
    RasterStageKernel fragment = [](
                                     Var<Metal4StencilVarying>) noexcept {
        return make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    };
    RasterKernel<decltype(vertex), decltype(fragment)> kernel{
        vertex, fragment};

    MeshFormat mesh_format;
    const VertexAttribute attributes[]{
        {VertexAttributeType::Position, PixelFormat::RGBA32F}};
    mesh_format.emplace_vertex_stream(attributes);
    auto shader = dc->device.compile(kernel, mesh_format);

    const std::array vertices{
        make_float4(-0.8f, -0.8f, 0.5f, 1.0f),
        make_float4(0.8f, -0.8f, 0.5f, 1.0f),
        make_float4(0.0f, 0.8f, 0.5f, 1.0f)};
    const std::array indices{0u, 1u, 2u};
    auto vertex_buffer = dc->device.create_buffer<float4>(vertices.size());
    auto index_buffer = dc->device.create_buffer<uint>(indices.size());
    auto render_target = dc->device.create_image<float>(
        PixelStorage::BYTE4, kWidth, kHeight, 1u, false, true);
    auto stream = dc->device.create_stream(StreamTag::GRAPHICS);
    auto raster = dc->device.extension<RasterExt>();
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << index_buffer.copy_from(luisa::span{indices})
           << synchronize();

    VertexBufferView vertex_view{vertex_buffer};
    auto draw = [&](auto &depth, const RasterState &state,
                    bool clear_color) noexcept {
        std::array<std::byte, kWidth * kHeight * 4u> pixels{};
        luisa::vector<RasterMesh> meshes;
        meshes.emplace_back(
            luisa::span<const VertexBufferView>{&vertex_view, 1u},
            index_buffer.view(), 1u, 0u);
        if (clear_color) {
            stream << raster->clear_render_target(
                render_target.view(),
                make_float4(0.0f, 0.0f, 0.0f, 1.0f));
        }
        stream << shader().draw(
                      std::move(meshes), mesh_format,
                      Viewport{0u, 0u, kWidth, kHeight},
                      state, &depth, render_target)
               << render_target.copy_to(luisa::span{pixels})
               << synchronize();
        return colored_pixel_count(pixels);
    };

    auto reset = [&](auto &depth, float value = 1.0f) noexcept {
        stream << depth.clear(value)
               << raster->clear_render_target(
                      render_target.view(),
                      make_float4(0.0f, 0.0f, 0.0f, 1.0f))
               << synchronize();
    };

    for (auto format : {DepthFormat::D24S8,
                        DepthFormat::D32S8A24}) {
        auto depth = dc->device.create_depth_buffer(
            format, make_uint2(kWidth, kHeight));

        // A passing fragment writes reference 1 with Replace. A subsequent
        // Equal test can only draw if the stencil value was really updated.
        reset(depth);
        static_cast<void>(draw(
            depth,
            raster_state(
                1u, Comparison::Always, Comparison::Always,
                StencilOp::Zero, StencilOp::Zero,
                StencilOp::Replace),
            false));
        auto pass_replace = draw(
            depth,
            raster_state(
                1u, Comparison::Equal, Comparison::Always),
            true);
        expect(pass_replace > 200u)
            << "stencil pass Replace produced " << pass_replace
            << " colored pixel(s) for format "
            << luisa::to_underlying(format);

        // A failing stencil comparison applies stencil_fail_op before color
        // or depth testing. Replace must make the follow-up Equal test pass.
        reset(depth);
        static_cast<void>(draw(
            depth,
            raster_state(
                1u, Comparison::Equal, Comparison::Always,
                StencilOp::Replace),
            false));
        auto stencil_fail_replace = draw(
            depth,
            raster_state(
                1u, Comparison::Equal, Comparison::Always),
            true);
        expect(stencil_fail_replace > 200u)
            << "stencil-fail Replace produced "
            << stencil_fail_replace
            << " colored pixel(s) for format "
            << luisa::to_underlying(format);

        // The fragment depth (0.5) fails Less against a cleared depth of 0.
        // depth_fail_op must still replace stencil with reference 1.
        reset(depth, 0.0f);
        static_cast<void>(draw(
            depth,
            raster_state(
                1u, Comparison::Always, Comparison::Less,
                StencilOp::Keep, StencilOp::Replace),
            false));
        auto depth_fail_replace = draw(
            depth,
            raster_state(
                1u, Comparison::Equal, Comparison::Always),
            true);
        expect(depth_fail_replace > 200u)
            << "depth-fail Replace produced "
            << depth_fail_replace
            << " colored pixel(s) for format "
            << luisa::to_underlying(format);

        // Zero must be observable after first priming stencil to one.
        reset(depth);
        static_cast<void>(draw(
            depth,
            raster_state(
                1u, Comparison::Always, Comparison::Always,
                StencilOp::Keep, StencilOp::Keep,
                StencilOp::Replace),
            false));
        static_cast<void>(draw(
            depth,
            raster_state(
                1u, Comparison::Equal, Comparison::Always,
                StencilOp::Keep, StencilOp::Keep,
                StencilOp::Zero),
            false));
        auto pass_zero = draw(
            depth,
            raster_state(
                0u, Comparison::Equal, Comparison::Always),
            true);
        expect(pass_zero > 200u)
            << "stencil pass Zero produced " << pass_zero
            << " colored pixel(s) for format "
            << luisa::to_underlying(format);

        // A zero write mask prevents Replace, so Equal(reference=1) rejects.
        reset(depth);
        static_cast<void>(draw(
            depth,
            raster_state(
                1u, Comparison::Always, Comparison::Always,
                StencilOp::Keep, StencilOp::Keep,
                StencilOp::Replace, 0xffu, 0u),
            false));
        auto write_masked = draw(
            depth,
            raster_state(
                1u, Comparison::Equal, Comparison::Always),
            true);
        expect(write_masked == 0u)
            << "zero stencil write mask left " << write_masked
            << " colored pixel(s) for format "
            << luisa::to_underlying(format);

        // With read mask zero, cleared stencil 0 and reference 1 both compare
        // as zero, so Equal must pass.
        reset(depth);
        auto read_masked = draw(
            depth,
            raster_state(
                1u, Comparison::Equal, Comparison::Always,
                StencilOp::Keep, StencilOp::Keep,
                StencilOp::Keep, 0u),
            false);
        expect(read_masked > 200u)
            << "zero stencil read mask produced " << read_masked
            << " colored pixel(s) for format "
            << luisa::to_underlying(format);
    }
    return 0;
}
