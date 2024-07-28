//
// Created by Mike Smith on 2021/6/23.
//
#include <stb/stb_image_write.h>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_info();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    // std::array vertices{
    //     float3(-0.5f, -0.5f, 0.0f),
    //     float3(0.5f, -0.5f, 0.0f),
    //     float3(0.0f, 0.5f, 0.0f)};
    // std::array indices{0u, 1u, 2u};

    // Callable linear_to_srgb = [](Var<float3> x) noexcept {
    //     return select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
    //                   12.92f * x,
    //                   x <= 0.00031308f);
    // };

    // Callable halton = [](UInt i, UInt b) noexcept {
    //     Float f = def(1.0f);
    //     Float invB = 1.0f / b;
    //     Float r = def(0.0f);
    //     $while (i > 0u) {
    //         f = f * invB;
    //         r = r + f * (i % b);
    //         i = i / b;
    //     };
    //     return r;
    // };

    // Callable tea = [](UInt v0, UInt v1) noexcept {
    //     UInt s0 = def(0u);
    //     for (uint n = 0u; n < 4u; n++) {
    //         s0 += 0x9e3779b9u;
    //         v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
    //         v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
    //     }
    //     return v0;
    // };

    // Callable rand = [&](UInt f, UInt2 p) noexcept {
    //     UInt i = tea(p.x, p.y) + f;
    //     Float rx = halton(i, 2u);
    //     Float ry = halton(i, 3u);
    //     return make_float2(rx, ry);
    // };

    // Kernel2D raytracing_kernel = [&](BufferFloat4 image, AccelVar accel, UInt frame_index) noexcept {
    //     UInt2 coord = dispatch_id().xy();
    //     Float2 p = (make_float2(coord) + rand(frame_index, coord)) /
    //                    make_float2(dispatch_size().xy()) * 2.0f -
    //                1.0f;
    //     Float3 color = def<float3>(0.3f, 0.5f, 0.7f);
    //     Var<Ray> ray = make_ray(
    //         make_float3(p * make_float2(1.0f, -1.0f), 1.0f),
    //         make_float3(0.0f, 0.0f, -1.0f));
    //     Var<TriangleHit> hit = accel.intersect(ray, {});
    //     $if (!hit->miss()) {
    //         constexpr float3 red = float3(1.0f, 0.0f, 0.0f);
    //         constexpr float3 green = float3(0.0f, 1.0f, 0.0f);
    //         constexpr float3 blue = float3(0.0f, 0.0f, 1.0f);
    //         color = triangle_interpolate(hit.bary, red, green, blue);
    //     };
    //     Float3 old = image.read(coord.y * dispatch_size_x() + coord.x).xyz();
    //     Float t = 1.0f / (frame_index + 1.0f);
    //     image.write(coord.y * dispatch_size_x() + coord.x, make_float4(lerp(old, color, t), 1.0f));
    // };

    // Kernel2D colorspace_kernel = [&](BufferFloat4 hdr_image, BufferUInt ldr_image) noexcept {
    //     UInt i = dispatch_y() * dispatch_size_x() + dispatch_x();
    //     Float3 hdr = hdr_image.read(i).xyz();
    //     UInt3 ldr = make_uint3(round(clamp(linear_to_srgb(hdr), 0.f, 1.f) * 255.0f));
    //     ldr_image.write(i, ldr.x | (ldr.y << 8u) | (ldr.z << 16u) | (255u << 24u));
    // };
    // Kernel1D set_transform_kernel = [&](AccelVar accel, Float4x4 matrix, UInt offset) noexcept {
    //     accel.set_instance_transform(dispatch_id().x + offset, matrix);
    // };
    auto buffer = device.create_buffer<float4>(1);
    float4x4 mat(
        float4(1, 2, 3, 4),
        float4(5, 6, 7, 8),
        float4(9, 10, 11, 12),
        float4(13, 14, 15, 16));
    float4 vec(666, 777, 888, 999);
    float4 r;
    Kernel1D set = [&](Float4x4 mat, Float4 vec) noexcept {
        set_block_size(32, 1, 1);
        buffer->write(0, mat * vec);
    };
    auto set_shader = device.compile(set);
    Stream stream = device.create_stream();
    stream << set_shader(mat, vec).dispatch(1) << buffer.copy_to(&r) << synchronize();
    LUISA_INFO("{}, {}, {}, {}", r.x, r.y, r.z, r.w);
}
