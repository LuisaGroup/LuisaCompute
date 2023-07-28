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

using namespace luisa;
using namespace luisa::compute;
struct v2p {
    float4 pos;
    float2 uv;
    float color;
};
LUISA_STRUCT(v2p, pos, uv, color){};
struct Vertex {
    std::array<float, 3> pos;
    std::array<float, 2> uv;
};
int main(int argc, char *argv[]) {

    RasterStageKernel vert = [&](Var<AppData> var, Float time) {
        Var<v2p> o;
        o.pos = make_float4(var.position, 1.f);
        $if(var.vertex_id >= 3) {
            o.pos.y += sin(time) * 0.1f;
            o.color = 0.5f;
        }
        $else {
            o.color = 0.f;
        };
        o.uv = var.uv[0];
        return o;
    };
    RasterStageKernel pixel = [&](Var<v2p> i, Float time) {
        // return make_float4(object_id().cast<float>() / 10.0f);
        return make_float4(i.uv, fract(cos(time) * 0.5f + 0.5f - i.color), 1.f);
    };
    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.1f));
    };
    RasterKernel<decltype(vert), decltype(pixel)> kernel{vert, pixel};
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1], nullptr);
    Shader2D<Image<float>> clear_shader = device.compile(clear_kernel);
    MeshFormat mesh_format;
    VertexAttribute attributes[] = {
        {VertexAttributeType::Position, VertexElementFormat::XYZ32Float},
        {VertexAttributeType::UV0, VertexElementFormat::XY32Float},
    };
    mesh_format.emplace_vertex_stream(attributes);
    static constexpr uint width = 1024;
    static constexpr uint height = 1024;
    DepthBuffer depth_buffer = device.create_depth_buffer(DepthFormat::D32, uint2(width, height));
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Window window{"Test raster", width, height};
    Swapchain swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        make_uint2(width, height),
        true, false, 2)};
    Image<float> out_img = device.create_image<float>(swap_chain.backend_storage(), width, height);
    PixelFormat img_format = out_img.format();
    RasterShader<float, float> shader = device.compile(
        kernel,
        mesh_format);
    Vertex vertices[6];
    vertices[0].pos = {-0.5f, 0.5f, 0.5f};
    vertices[0].uv = {0.0f, 0.0f};
    vertices[1].pos = {0.5f, 0.5f, 0.5f};
    vertices[1].uv = {1.0f, 0.0f};
    vertices[2].pos = {0.0f, -0.5f, 0.5f};
    vertices[2].uv = {0.0f, 1.0f};

    vertices[3].pos = {-0.7f, 0.5f, 0.2f};
    vertices[3].uv = {0.0f, 0.0f};
    vertices[4].pos = {0.5f, 0.2f, 0.8f};
    vertices[4].uv = {1.0f, 0.0f};
    vertices[5].pos = {0.2f, -0.5f, 0.3f};
    vertices[5].uv = {0.0f, 1.0f};

    Buffer<Vertex> vert_buffer = device.create_buffer<Vertex>(6);
    Buffer<uint> idx_buffer = device.create_buffer<uint>(6);
    uint indices[6] = {
        0, 1, 2, 3, 4, 5};
    stream << vert_buffer.copy_from(vertices)
           << idx_buffer.copy_from(indices);
    VertexBufferView vert_buffer_view{vert_buffer};
    Clock clock;
    clock.tic();
    luisa::vector<RasterMesh> meshes;
    RasterState state{
        .cull_mode = CullMode::None,
        .depth_state = DepthState{
            .enable_depth = true,
            .comparison = Comparison::Less,
            .write = true}};
    while (!window.should_close()) {
        float time = clock.toc() / 1000.0f;
        // add triangle mesh
        meshes.emplace_back(luisa::span<VertexBufferView const>{&vert_buffer_view, 1}, idx_buffer, 1, 0);
        stream
            // clear depth buffer
            << depth_buffer.clear(1.f)
            << clear_shader(out_img).dispatch(width, height)
            << shader(time, time * 5).draw(std::move(meshes), Viewport{}, state, &depth_buffer, out_img)
            << swap_chain.present(out_img);
        window.poll_events();
    }
    stream << synchronize();
    return 0;
}
