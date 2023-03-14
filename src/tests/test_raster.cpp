#include <runtime/rhi/command.h>
#include <runtime/raster/raster_shader.h>
#include <dsl/raster/raster_kernel.h>
#include <core/logging.h>
#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <runtime/stream.h>
#include <runtime/image.h>
#include <runtime/raster/raster_scene.h>
#include <runtime/raster/raster_state.h>
#include <gui/window.h>
#include <core/clock.h>
using namespace luisa;
using namespace luisa::compute;
struct v2p {
    float4 pos;
    float2 uv;
};
LUISA_STRUCT(v2p, pos, uv){};
struct Vertex {
    std::array<float, 3> pos;
    std::array<float, 2> uv;
};
int main(int argc, char *argv[]) {

    Callable vert = [&](Var<AppData> var, Float time) {
        Var<v2p> o;
        o.pos = make_float4(var.position, 1.f);
        o.pos.y += sin(time) * 0.1f;
        o.uv = var.uv[0];
        return o;
    };
    Callable pixel = [&](Var<v2p> i, Float time) {
        return make_float4(i.uv, cos(time) * 0.5f + 0.5f, 1.f);
    };
    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.1f));
    };
    RasterKernel<decltype(vert), decltype(pixel)> kernel{vert, pixel};
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    auto clear_shader = device.compile(clear_kernel);
    MeshFormat mesh_format;
    VertexAttribute attributes[] = {
        {VertexAttributeType::Position, VertexElementFormat::XYZ32Float},
        {VertexAttributeType::UV0, VertexElementFormat::XY32Float},
    };
    mesh_format.emplace_vertex_stream(attributes);
    RasterState state{.cull_mode = CullMode::None};
    static constexpr uint32_t width = 1024;
    static constexpr uint32_t height = 1024;
    auto out_img = device.create_image<float>(PixelStorage::BYTE4, width, height);
    auto img_format = out_img.format();
    auto shader = device.compile(
        kernel,
        mesh_format,
        state,
        {&img_format, 1},
        DepthFormat::None);
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    Window window{"Test raster", width, height, false};
    Window window2{"Test raster 2", width, height, false};
    auto swap_chain = device.create_swapchain(
        window.native_handle(),
        stream,
        make_uint2(width, height),
        true, false, 2);
    auto swap_chain2 = device.create_swapchain(
        window2.native_handle(),
        stream,
        make_uint2(width, height),
        true, false, 2);
    auto vert_buffer = device.create_buffer<Vertex>(3);
    auto idx_buffer = device.create_buffer<uint32_t>(3);
    Vertex vertices[3];
    vertices[0].pos = {-0.5f, -0.5f, 0.5f};
    vertices[0].uv = {0.0f, 0.0f};
    vertices[1].pos = {0.5f, -0.5f, 0.5f};
    vertices[1].uv = {1.0f, 0.0f};
    vertices[2].pos = {0.0f, 0.5f, 0.5f};
    vertices[2].uv = {0.0f, 1.0f};
    uint32_t indices[3] = {
        0, 1, 2};
    stream << vert_buffer.copy_from(vertices)
           << idx_buffer.copy_from(indices);
    VertexBufferView vert_buffer_view{vert_buffer};
    Clock clock;
    clock.tic();
    luisa::vector<RasterMesh> meshes;
    while (!window.should_close() && !window2.should_close()) {
        auto time = clock.toc() / 1000.0f;
        // add triangle mesh
        meshes.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 0);
        stream
            << clear_shader(out_img).dispatch(width, height)
            << shader(time, time * 5).draw(std::move(meshes), {}, nullptr, out_img)
            << swap_chain.present(out_img)
            << swap_chain2.present(out_img);
        window.pool_event();
        window2.pool_event();
    }
    return 0;
}