#include <iostream>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <stb/stb_image_write.h>
#include <luisa/gui/window.h>
#include <luisa/ast/ast2json.h>

#include "common/cornell_box.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "common/tiny_obj_loader.h"

// #define CLANG_CXX

using namespace luisa;
using namespace luisa::compute;

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    // load the Cornell Box scene
    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromString(obj_string, "", obj_reader_config)) {
        luisa::string_view error_message = "unknown error.";
        if (auto &&e = obj_reader.Error(); !e.empty()) { error_message = e; }
        LUISA_ERROR_WITH_LOCATION("Failed to load OBJ file: {}", error_message);
    }
    if (auto &&e = obj_reader.Warning(); !e.empty()) {
        LUISA_WARNING_WITH_LOCATION("{}", e);
    }

    auto &&p = obj_reader.GetAttrib().vertices;
    luisa::vector<float3> vertices;
    vertices.reserve(p.size() / 3u);
    for (uint i = 0u; i < p.size(); i += 3u) {
        vertices.emplace_back(make_float3(
            p[i + 0u], p[i + 1u], p[i + 2u]));
    }
    LUISA_INFO(
        "Loaded mesh with {} shape(s) and {} vertices.",
        obj_reader.GetShapes().size(), vertices.size());

    BindlessArray heap = device.create_bindless_array();
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.copy_from(vertices.data());
    luisa::vector<Mesh> meshes;
    luisa::vector<Buffer<Triangle>> triangle_buffers;
    for (auto &&shape : obj_reader.GetShapes()) {
        uint index = static_cast<uint>(meshes.size());
        std::vector<tinyobj::index_t> const &t = shape.mesh.indices;
        uint triangle_count = t.size() / 3u;
        LUISA_INFO(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, index, triangle_count);
        luisa::vector<uint> indices;
        indices.reserve(t.size());
        for (tinyobj::index_t i : t) { indices.emplace_back(i.vertex_index); }
        Buffer<Triangle> &triangle_buffer = triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
        Mesh &mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
        heap.emplace_on_update(index, triangle_buffer);
        stream << triangle_buffer.copy_from(indices.data())
               << mesh.build();
    }

    Accel accel = device.create_accel({});
    for (Mesh &m : meshes) {
        accel.emplace_back(m, make_float4x4(1.0f));
    }
    stream << heap.update()
           << accel.build()
           << synchronize();

    float3 mats[] = {
        make_float3(0.725f, 0.710f, 0.680f),// floor
        make_float3(0.725f, 0.710f, 0.680f),// ceiling
        make_float3(0.725f, 0.710f, 0.680f),// back wall
        make_float3(0.140f, 0.450f, 0.091f),// right wall
        make_float3(0.630f, 0.065f, 0.050f),// left wall
        make_float3(0.725f, 0.710f, 0.680f),// short box
        make_float3(0.725f, 0.710f, 0.680f),// tall box
        make_float3(0.000f, 0.000f, 0.000f),// light
    };
    auto materials = device.create_buffer<float3>(8);
    stream << materials.copy_from(mats);

    Callable linear_to_srgb = [&](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D make_sampler_kernel = [&](ImageUInt seed_image) noexcept {
        UInt2 p = dispatch_id().xy();
        UInt state = tea(p.x, p.y);
        seed_image.write(p, make_uint4(state));
    };
    auto spp_per_dispatch = device.backend_name() == "metal" || device.backend_name() == "cpu" ? 1u : 64u;

    Kernel2D accumulate_kernel = [&](ImageFloat accum_image, ImageFloat curr_image) noexcept {
        UInt2 p = dispatch_id().xy();
        Float4 accum = accum_image.read(p);
        Float3 curr = curr_image.read(p).xyz();
        accum_image.write(p, accum + make_float4(curr, 1.f));
    };

    Callable aces_tonemapping = [](Float3 x) noexcept {
        static constexpr float a = 2.51f;
        static constexpr float b = 0.03f;
        static constexpr float c = 2.43f;
        static constexpr float d = 0.59f;
        static constexpr float e = 0.14f;
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    };

    Kernel2D clear_kernel = [](ImageFloat image) noexcept {
        image.write(dispatch_id().xy(), make_float4(0.0f));
    };

    Kernel2D hdr2ldr_kernel = [&](ImageFloat hdr_image, ImageFloat ldr_image, Float scale, Bool is_hdr) noexcept {
        UInt2 coord = dispatch_id().xy();
        Float4 hdr = hdr_image.read(coord);
        Float3 ldr = hdr.xyz() / hdr.w * scale;
        $if (!is_hdr) {
            ldr = linear_to_srgb(ldr);
        };
        ldr_image.write(coord, make_float4(ldr, 1.0f));
    };

    ShaderOption o{.enable_debug_info = false};
    auto clear_shader = device.compile(clear_kernel, o);
    auto hdr2ldr_shader = device.compile(hdr2ldr_kernel, o);
    auto accumulate_shader = device.compile(accumulate_kernel, o);
    auto raytracing_shader = device.load_shader<2, Image<float>, Image<uint>, Accel, BindlessArray, Buffer<float3>, Buffer<float3>, uint2>("test.bin");
    auto make_sampler_shader = device.compile(make_sampler_kernel, o);

    static constexpr uint2 resolution = make_uint2(1024u);
    Image<float> framebuffer = device.create_image<float>(PixelStorage::HALF4, resolution);
    Image<float> accum_image = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    luisa::vector<std::array<uint8_t, 4u>> host_image(resolution.x * resolution.y);
    CommandList cmd_list;
    Image<uint> seed_image = device.create_image<uint>(PixelStorage::INT1, resolution);
    cmd_list << clear_shader(accum_image).dispatch(resolution)
             << make_sampler_shader(seed_image).dispatch(resolution);

    Window window{"path tracing", resolution};
    Swapchain swap_chain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = make_uint2(resolution),
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 2,
        });
    Image<float> ldr_image = device.create_image<float>(swap_chain.backend_storage(), resolution);
    double last_time = 0.0;
    uint frame_count = 0u;
    Clock clock;

    while (!window.should_close()) {
        cmd_list << raytracing_shader(framebuffer, seed_image, accel, heap, vertex_buffer, materials, resolution)
                        .dispatch(resolution)
                 << accumulate_shader(accum_image, framebuffer)
                        .dispatch(resolution);
        cmd_list << hdr2ldr_shader(accum_image, ldr_image, 1.0f, swap_chain.backend_storage() != PixelStorage::BYTE4).dispatch(resolution);
        stream << cmd_list.commit()
               << swap_chain.present(ldr_image) << synchronize();
        window.poll_events();
        double dt = clock.toc() - last_time;
        last_time = clock.toc();
        frame_count += spp_per_dispatch;
        LUISA_INFO("spp: {}, time: {} ms, spp/s: {}",
                   frame_count, dt, spp_per_dispatch / dt * 1000);
    }
    stream
        << ldr_image.copy_to(host_image.data())
        << synchronize();

    LUISA_INFO("FPS: {}", frame_count / clock.toc() * 1000);
    stbi_write_png("test_path_tracing.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}
