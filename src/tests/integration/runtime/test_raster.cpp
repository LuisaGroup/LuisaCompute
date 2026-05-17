#include "ut/ut.hpp"
#include "test_device.h"

#include "reference_image.h"

#include <filesystem>

#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/dsl/raster/raster_kernel.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/gui/window.h>
#include <luisa/core/clock.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/backends/ext/raster_ext.hpp>
using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;
struct v2p {
    float4 pos;
    float2 uv;
    float4 color;
};
LUISA_STRUCT(v2p, pos, uv, color) {};
struct Vertex {
    float3 pos;
    float3 normal;
    float4 tangent;
    float2 uv1;
    uint color;
};
void test_raster(Device &device) {
    auto argv = boost::ut::detail::cfg::largv;

    // RasterStageKernel vert = [&](Var<AppData> var, Float time) {
    //     Var<v2p> o;
    //     o.pos = make_float4(var.position, 1.f);
    //     $if (var.vertex_id >= 3) {
    //         o.pos.y += sin(time) * 0.1f;
    //         o.color = make_float4(0.3f, 0.6f, 0.7f, 1.0f);
    //     }
    //     $else {
    //         o.color = make_float4(0.7f, 0.6f, 0.3f, 1.0f);
    //     };
    //     o.uv = float2(0.5);
    //     return o;
    // };
    // RasterStageKernel pixel = [&](Var<v2p> i, Float time) {
    //     return i.color;
    // };
    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.1f));
    };
    // RasterKernel<decltype(vert), decltype(pixel)> kernel{vert, pixel};
    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    static constexpr uint width = 1024;
    static constexpr uint height = 1024;
    auto shader = device.load_raster_shader<float, float>(luisa::format("test_{}.bin", argv[1]));

    DepthBuffer depth_buffer = device.create_depth_buffer(DepthFormat::D32, uint2(width, height));
    auto clear_shader = device.compile(clear_kernel);
    MeshFormat mesh_format;
    VertexAttribute attributes[] = {
        {VertexAttributeType::Position, PixelFormat::RGBA32F},
        {VertexAttributeType::Normal, PixelFormat::RGBA32F},
        {VertexAttributeType::Tangent, PixelFormat::RGBA32F},
        {VertexAttributeType::UV0, PixelFormat::RG32F},
        {VertexAttributeType::Color, PixelFormat::RG32F},
    };
    mesh_format.emplace_vertex_stream(attributes);
    auto ref_dir = luisa::test::find_reference_dir(std::filesystem::path{argv[0]}.parent_path());

    Vertex vertices[6];
    vertices[0].pos = {-0.5f, 0.5f, 0.5f};
    vertices[1].pos = {0.5f, 0.5f, 0.5f};
    vertices[2].pos = {0.0f, -0.5f, 0.5f};

    vertices[3].pos = {-0.7f, 0.5f, 0.2f};
    vertices[4].pos = {0.5f, 0.2f, 0.8f};
    vertices[5].pos = {0.2f, -0.5f, 0.3f};

    Buffer<Vertex> vert_buffer = device.create_buffer<Vertex>(6);
    Buffer<uint> idx_buffer = device.create_buffer<uint>(3);
    uint indices[3] = {
        0, 1, 2};
    stream << vert_buffer.copy_from(luisa::span{vertices, std::size(vertices)})
           << idx_buffer.copy_from(luisa::span{indices, std::size(indices)});
    VertexBufferView vert_buffer_view{vert_buffer};
    Clock clock;
    clock.tic();
    RasterState state{
        .cull_mode = CullMode::None,
        .depth_state = DepthState{
            .enable_depth = true,
            .comparison = Comparison::Less,
            .write = true},
        .conservative = true};
    if (!opts.offline) {
        Window window{"Test raster", width, height};
        Swapchain swap_chain = device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window.native_display(),
                .window = window.native_handle(),
                .size = make_uint2(width, height),
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 2,
            });
        Image<float> out_img = device.create_image<float>(swap_chain.backend_storage(), width, height, 1, false, true);
        PixelFormat img_format = out_img.format();
        (void)img_format;
        while (!window.should_close()) {
            float time = clock.toc() / 1000.0f;
            // add triangle mesh
            luisa::vector<RasterMesh> meshes;
            meshes.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 114514);
            meshes.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 1919810, 3);
            stream
                // clear depth buffer
                << clear_shader(out_img).dispatch(width, height)
                << depth_buffer.clear(1.0)
                << shader(time, time * 5).draw(std::move(meshes), mesh_format, Viewport{0, 0, width, height}, state, &depth_buffer, out_img)
                << swap_chain.present(out_img);
            window.poll_events();
        }
        stream << synchronize();
        return;
    } else {
        Image<float> out_img = device.create_image<float>(PixelStorage::BYTE4, width, height, 1, false, true);
        luisa::vector<std::byte> pixels(out_img.view().size_bytes());
        luisa::vector<RasterMesh> meshes;
        meshes.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 114514);
        meshes.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 1919810, 3);
        stream
            << clear_shader(out_img).dispatch(width, height)
            << depth_buffer.clear(1.0)
            << shader(0.0f, 0.0f).draw(std::move(meshes), mesh_format, Viewport{0, 0, width, height}, state, &depth_buffer, out_img)
            << out_img.copy_to(luisa::span{pixels})
            << synchronize();
        auto result = luisa::test::save_and_compare(
            reinterpret_cast<const uint8_t *>(pixels.data()), static_cast<int>(width), static_cast<int>(height), 4,
            "test_raster", opts.output_dir, ref_dir, opts.update_reference);
        LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
        if (!result.passed) {
            LUISA_ERROR("Reference comparison failed for test_raster: {}", result.message);
            boost::ut::expect(false) << result.message;
            return;
        }
        return;
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_raster(device);
}
