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
int main(int argc, char *argv[]) {

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
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1], nullptr);
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
    static constexpr uint width = 1024;
    static constexpr uint height = 1024;
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
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
    Image<float> out_img = device.create_image<float>(swap_chain.backend_storage(), width, height);
    PixelFormat img_format = out_img.format();
    DepthBuffer depth_buffer = device.create_depth_buffer(DepthFormat::D32, uint2(width, height));
    auto shader = device.load_raster_shader<float, float>("test.bin");
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
    stream << vert_buffer.copy_from(vertices)
           << idx_buffer.copy_from(indices);
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
    return 0;
}
