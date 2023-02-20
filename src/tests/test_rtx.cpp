//
// Created by Mike Smith on 2021/6/23.
//
#include <stb/stb_image_write.h>

#include <core/clock.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <runtime/rtx/accel.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_info();

    Context context{argv[0]};
    if(argc <= 1){
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);

    std::array vertices{
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.0f, 0.5f, 0.0f)};
    std::array indices{0u, 1u, 2u};

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                      12.92f * x,
                      x <= 0.00031308f);
    };

    Callable halton = [](UInt i, UInt b) noexcept {
        auto f = def(1.0f);
        auto invB = 1.0f / b;
        auto r = def(0.0f);
        $while(i > 0u) {
            f = f * invB;
            r = r + f * (i % b);
            i = i / b;
        };
        return r;
    };

    Callable tea = [](UInt v0, UInt v1) noexcept {
        auto s0 = def(0u);
        for (auto n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Callable rand = [&](UInt f, UInt2 p) noexcept {
        auto i = tea(p.x, p.y) + f;
        auto rx = halton(i, 2u);
        auto ry = halton(i, 3u);
        return make_float2(rx, ry);
    };

    Kernel2D raytracing_kernel = [&](BufferFloat4 image, AccelVar accel, UInt frame_index) noexcept {
        auto coord = dispatch_id().xy();
        auto p = (make_float2(coord) + rand(frame_index, coord)) /
                     make_float2(dispatch_size().xy()) * 2.0f -
                 1.0f;
        auto color = def<float3>(0.3f, 0.5f, 0.7f);
        auto ray = make_ray(
            make_float3(p * make_float2(1.0f, -1.0f), 1.0f),
            make_float3(0.0f, 0.0f, -1.0f));
        auto hit = accel.trace_closest(ray);
        $if(!hit->miss()) {
            constexpr auto red = float3(1.0f, 0.0f, 0.0f);
            constexpr auto green = float3(0.0f, 1.0f, 0.0f);
            constexpr auto blue = float3(0.0f, 0.0f, 1.0f);
            color = hit->interpolate(red, green, blue);
        };
        auto old = image.read(coord.y * dispatch_size_x() + coord.x).xyz();
        auto t = 1.0f / (frame_index + 1.0f);
        image.write(coord.y * dispatch_size_x() + coord.x, make_float4(lerp(old, color, t), 1.0f));
    };

    Kernel2D colorspace_kernel = [&](BufferFloat4 hdr_image, BufferUInt ldr_image) noexcept {
        auto i = dispatch_y() * dispatch_size_x() + dispatch_x();
        auto hdr = hdr_image.read(i).xyz();
        auto ldr = make_uint3(round(saturate(linear_to_srgb(hdr)) * 255.0f));
        ldr_image.write(i, ldr.x | (ldr.y << 8u) | (ldr.z << 16u) | (255u << 24u));
    };
    auto stream = device.create_stream();
    auto vertex_buffer = device.create_buffer<float3>(3u);
    auto triangle_buffer = device.create_buffer<Triangle>(1u);
    stream << vertex_buffer.copy_from(vertices.data())
           << triangle_buffer.copy_from(indices.data());

    auto accel = device.create_accel();
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    accel.emplace_back(mesh, scaling(1.5f));
    accel.emplace_back(mesh, translation(float3(-0.25f, 0.0f, 0.1f)) *
                                 rotation(float3(0.0f, 0.0f, 1.0f), 0.5f));
    stream << mesh.build()
           << accel.build();

    auto colorspace_shader = device.compile(colorspace_kernel);
    auto raytracing_shader = device.compile(raytracing_kernel);

    static constexpr auto width = 512u;
    static constexpr auto height = 512u;
    auto hdr_image = device.create_buffer<float4>(width * height);
    auto ldr_image = device.create_buffer<uint>(width * height);
    std::vector<uint8_t> pixels(width * height * 4u);

    Clock clock;
    clock.tic();
    static constexpr auto spp = 1024u;
    for (auto i = 0u; i < spp; i++) {
        auto t = static_cast<float>(i) * (1.0f / spp);
        vertices[2].y = 0.5f - 0.2f * t;
        auto m = translation(float3(-0.25f + t * 0.15f, 0.0f, 0.1f)) *
                 rotation(float3(0.0f, 0.0f, 1.0f), 0.5f + t * 0.5f);
        accel.set_transform_on_update(1u, m);
        stream << vertex_buffer.copy_from(vertices.data())
               << mesh.build()
               << accel.build()
               << raytracing_shader(hdr_image, accel, i).dispatch(width, height);
        if (i == 511u) {
            auto mm = translation(make_float3(0.0f, 0.0f, 0.3f)) *
                      rotation(make_float3(0.0f, 0.0f, 1.0f), radians(180.0f));
            accel.emplace_back(mesh, mm, true);
            stream << accel.build();
        }
    }
    stream << colorspace_shader(hdr_image, ldr_image).dispatch(width, height)
           << ldr_image.copy_to(pixels.data())
           << synchronize();
    auto time = clock.toc();
    LUISA_INFO("Time: {} ms", time);
    stbi_write_png("test_rtx.png", width, height, 4, pixels.data(), 0);
}