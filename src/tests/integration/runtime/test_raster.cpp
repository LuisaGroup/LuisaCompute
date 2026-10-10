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
        // Render into a nonzero color mip whose extent matches the depth
        // attachment. This exercises attachment-view levels, framebuffer
        // extent derivation, and per-mip layout tracking without changing the
        // reference image dimensions.
        Image<float> out_img = device.create_image<float>(
            PixelStorage::BYTE4, width * 2u, height * 2u,
            2u, false, true);
        auto out_view = out_img.view(1u);
        luisa::vector<std::byte> pixels(out_view.size_bytes());
        luisa::vector<RasterMesh> meshes;
        meshes.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 114514);
        meshes.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 1919810, 3);
        stream
            << clear_shader(out_view).dispatch(width, height)
            << depth_buffer.clear(1.0)
            << shader(0.0f, 0.0f).draw(std::move(meshes), mesh_format, Viewport{0, 0, width, height}, state, &depth_buffer, out_view)
            << out_view.copy_to(luisa::span{pixels})
            << synchronize();
        if (opts.compare_path) {
            auto result = luisa::test::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(pixels.data()), static_cast<int>(width), static_cast<int>(height), 4,
                *opts.compare_path);
            LUISA_INFO("Reference comparison [test_raster]: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) {
            boost::ut::expect(static_cast<bool>(result.passed)) << result.message;
            return;
        }
        }

        // Robustness probes for the raster backends' pipeline caches:
        // (1) The same shader and raster state drawn with two different
        //     MeshFormats (describing the same triangle through different
        //     vertex buffer layouts) must produce identical results. The DX
        //     backend used to key its PSO cache without the mesh format, so
        //     the second draw reused the first input layout.
        // (2) The same shader and raster state drawn into two different
        //     color attachment formats must both render correctly. The
        //     Vulkan backend used to key its pipeline cache without the
        //     attachment formats, so the second draw reused a render pass
        //     created for the first format.
        {
            struct PaddedVertex {
                float4 pad;
                float4 pos;
            };
            static_assert(sizeof(PaddedVertex) == 32u);
            PaddedVertex padded[6];
            for (size_t i = 0; i < 6u; ++i) {
                padded[i].pad = make_float4(0.f);
                padded[i].pos = make_float4(vertices[i].pos, 1.f);
            }
            Buffer<PaddedVertex> padded_buffer = device.create_buffer<PaddedVertex>(6);
            stream << padded_buffer.copy_from(luisa::span{padded, std::size(padded)});
            VertexBufferView padded_view{padded_buffer};

            // Mesh format A: position at offset 0 of the plain Vertex buffer.
            MeshFormat format_a;
            VertexAttribute attributes_a[] = {
                {VertexAttributeType::Position, PixelFormat::RGBA32F},
                {VertexAttributeType::Normal, PixelFormat::RGBA32F}};
            format_a.emplace_vertex_stream(attributes_a);
            // Mesh format B: position at offset 16 of the padded buffer,
            // behind the NORMAL attribute.
            MeshFormat format_b;
            VertexAttribute attributes_b[] = {
                {VertexAttributeType::Normal, PixelFormat::RGBA32F},
                {VertexAttributeType::Position, PixelFormat::RGBA32F}};
            format_b.emplace_vertex_stream(attributes_b);

            auto img_a = device.create_image<float>(PixelStorage::BYTE4, width, height, 1, false, true);
            auto img_b = device.create_image<float>(PixelStorage::BYTE4, width, height, 1, false, true);
            luisa::vector<RasterMesh> meshes_a;
            meshes_a.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 114514);
            luisa::vector<RasterMesh> meshes_b;
            meshes_b.emplace_back(luisa::span<VertexBufferView const>{&padded_view, 1}, idx_buffer, 1, 114514);
            luisa::vector<std::byte> pixels_a(pixels.size());
            luisa::vector<std::byte> pixels_b(pixels.size());
            stream
                << clear_shader(img_a).dispatch(width, height)
                << clear_shader(img_b).dispatch(width, height)
                << depth_buffer.clear(1.0)
                << shader(0.0f, 0.0f).draw(std::move(meshes_a), format_a, Viewport{0, 0, width, height}, state, &depth_buffer, img_a)
                << depth_buffer.clear(1.0)
                << shader(0.0f, 0.0f).draw(std::move(meshes_b), format_b, Viewport{0, 0, width, height}, state, &depth_buffer, img_b)
                << img_a.copy_to(luisa::span{pixels_a})
                << img_b.copy_to(luisa::span{pixels_b})
                << synchronize();
            size_t mismatch = 0;
            for (size_t i = 0; i < pixels_a.size(); ++i) {
                mismatch += (pixels_a[i] != pixels_b[i]) ? 1u : 0u;
            }
            boost::ut::expect(mismatch == 0u)
                << luisa::format("Mesh-format probe: {} of {} bytes differ between the two mesh formats.", mismatch, pixels_a.size());
            if (mismatch != 0u) {
                stbi_write_png("test_raster_probe_a.png", width, height, 4, pixels_a.data(), width * 4);
                stbi_write_png("test_raster_probe_b.png", width, height, 4, pixels_b.data(), width * 4);
                return;
            }

            // Probe 2: same mesh/state, different color attachment format.
            auto img_float = device.create_image<float>(PixelStorage::FLOAT4, width, height, 1, false, true);
            luisa::vector<float> float_pixels(static_cast<size_t>(width) * height * 4u);
            luisa::vector<RasterMesh> meshes_c;
            meshes_c.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 114514);
            stream
                << clear_shader(img_float).dispatch(width, height)
                << depth_buffer.clear(1.0)
                << shader(0.0f, 0.0f).draw(std::move(meshes_c), mesh_format, Viewport{0, 0, width, height}, state, &depth_buffer, img_float)
                << img_float.copy_to(luisa::span{float_pixels})
                << synchronize();
            size_t float_mismatch = 0;
            for (size_t p = 0; p < static_cast<size_t>(width) * height; ++p) {
                for (uint c = 0; c < 4u; ++c) {
                    auto v = float_pixels[p * 4u + c];
                    v = std::clamp(v, 0.f, 1.f);
                    auto as_byte = static_cast<uint8_t>(v * 255.f + 0.5f);
                    auto diff = std::abs(static_cast<int>(as_byte) - static_cast<int>(reinterpret_cast<const uint8_t *>(pixels_a.data())[p * 4u + c]));
                    float_mismatch += (diff > 1) ? 1u : 0u;
                }
            }
            boost::ut::expect(float_mismatch == 0u)
                << luisa::format("Attachment-format probe: {} of {} channel values differ between BYTE4 and FLOAT4 attachments.", float_mismatch, pixels_a.size());
            if (float_mismatch != 0u) {
                luisa::vector<uint8_t> float_as_bytes(float_pixels.size());
                for (size_t i = 0; i < float_pixels.size(); ++i) {
                    auto v = std::clamp(float_pixels[i], 0.f, 1.f);
                    float_as_bytes[i] = static_cast<uint8_t>(v * 255.f + 0.5f);
                }
                stbi_write_png("test_raster_probe_float.png", width, height, 4, float_as_bytes.data(), width * 4);
            }

            // Probe 3: blending that keeps the previous attachment contents
            // (dst factor Zero) must render the same image as the opaque
            // draw. This exercises the blend render-pass path with a LOAD
            // op and a preserved initial layout.
            auto blend_state = state;
            blend_state.blend_state.enable_blend = true;
            blend_state.blend_state.prim_op = BlendWeight::One;
            blend_state.blend_state.img_op = BlendWeight::Zero;
            auto img_blend = device.create_image<float>(PixelStorage::BYTE4, width, height, 1, false, true);
            luisa::vector<std::byte> pixels_blend(pixels.size());
            luisa::vector<RasterMesh> meshes_d;
            meshes_d.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 114514);
            stream
                << clear_shader(img_blend).dispatch(width, height)
                << depth_buffer.clear(1.0)
                << shader(0.0f, 0.0f).draw(std::move(meshes_d), mesh_format, Viewport{0, 0, width, height}, blend_state, &depth_buffer, img_blend)
                << img_blend.copy_to(luisa::span{pixels_blend})
                << synchronize();
            size_t blend_mismatch = 0;
            for (size_t i = 0; i < pixels_a.size(); ++i) {
                blend_mismatch += (pixels_blend[i] != pixels_a[i]) ? 1u : 0u;
            }
            boost::ut::expect(blend_mismatch == 0u)
                << luisa::format("Blend probe: {} of {} bytes differ between the blended and the opaque draw.", blend_mismatch, pixels_a.size());
            if (blend_mismatch != 0u) {
                stbi_write_png("test_raster_probe_blend.png", width, height, 4, pixels_blend.data(), width * 4);
            }
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
