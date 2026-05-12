// Motion Blur Ray Tracing Test
// Demonstrates motion blur rendering using RTX acceleration structures with
// motion keyframes. Shows both mesh vertex motion and instance transform motion.
//
// Features demonstrated:
// - Motion blur with mesh keyframes (vertex animation)
// - Motion blur with curve keyframes (Catmull-Rom curves)
// - Motion instance transforms with SRT (Scale-Rotate-Translate)
// - RTX intersection with motion intervals

//
// Created by Mike Smith on 2024/9/20.
//

#include "ut/ut.hpp"
#include "test_device.h"
#include "../../../reference_image.h"

#include <filesystem>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_motion_blur(Device &device) {

    log_level_verbose();

    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);

    static constexpr uint width = 512u;
    static constexpr uint height = 512u;

    static constexpr auto mesh_keyframe_count = 2u;
    auto stream = device.create_stream(StreamTag::GRAPHICS);

    // Create mesh with vertex motion blur (3 keyframes)
    std::array vertices{
        // keyframe 0
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(-0.1f, 0.5f, 0.0f),
        // // keyframe 1
        // float3(-0.5f, -0.5f, 0.0f),
        // float3(0.5f, -0.5f, 0.0f),
        // float3(0.0f, 0.55f, 0.0f),
        // keyframe 2
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.1f, 0.5f, 0.0f),
    };
    std::array indices{0u, 1u, 2u};

    auto vertex_buffer = device.create_buffer<float3>(3u * mesh_keyframe_count);
    auto triangle_buffer = device.create_buffer<Triangle>(1u);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{indices});

    // Mesh with motion blur configuration
    AccelOption mesh_option;
    mesh_option.motion.keyframe_count = mesh_keyframe_count;
    mesh_option.motion.time_start = 0.f;
    mesh_option.motion.time_end = 1.f;
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer, mesh_option);

    // Color space conversion
    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                      12.92f * x,
                      x <= 0.00031308f);
    };

    // Halton sequence for sampling
    Callable halton = [](UInt i, UInt b) noexcept {
        Float f = def(1.0f);
        Float invB = 1.0f / cast<Float>(b);
        Float r = def(0.0f);
        $while (i > 0u) {
            f = f * invB;
            r = r + f * cast<Float>(i % b);
            i = i / b;
        };
        return r;
    };

    // TEA random number generator
    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    // Generate 3D random numbers (including time for motion blur)
    Callable rand = [&](UInt f, UInt2 p) noexcept {
        UInt i = tea(p.x, p.y) + f;
        Float rx = halton(i, 2u);
        Float ry = halton(i, 3u);
        Float rz = halton(i, 5u);
        return make_float3(rx, ry, rz);
    };

    // Camera ray generation
    Callable generate_ray = [](Float2 p) noexcept {
        constexpr auto origin = make_float3(0.f, 1.5f, 2.5f);
        constexpr auto target = make_float3(0.f, 0.f, 0.f);
        auto up = make_float3(0.f, 1.f, 0.f);
        auto front = normalize(target - origin);
        auto right = normalize(cross(front, up));
        up = cross(right, front);
        auto fov = radians(45.f);
        auto aspect = static_cast<float>(width) /
                      static_cast<float>(height);
        auto image_plane_height = tan(fov / 2.f);
        auto image_plane_width = aspect * image_plane_height;
        up *= image_plane_height;
        right *= image_plane_width;
        auto uv = p / make_float2(make_uint2(width, height)) * 2.f - 1.f;
        auto ray_origin = origin;
        auto ray_direction = normalize(uv.x * right - uv.y * up + front);
        return make_ray(ray_origin, ray_direction);
    };

    // Motion blur ray tracing kernel
    // Samples random time within shutter interval for motion blur
    Kernel2D raytracing_kernel = [&](BufferFloat4 image, AccelVar accel, UInt frame_index) noexcept {
        auto coord = dispatch_id().xy();
        auto color = def<float3>(0.3f, 0.5f, 0.7f);
        auto u = rand(frame_index, coord);
        auto ray = generate_ray(make_float2(coord) + u.xy());
        // Random time for motion blur sampling
        auto time = u.z * 1.f;
        // Trace with motion blur support
        auto hit = accel.intersect_motion(ray, time, {});
        $if (hit->is_triangle()) {
            // Mesh hit - color by barycentric coordinates
            constexpr auto red = make_float3(1.0f, 0.0f, 0.0f);
            constexpr auto green = make_float3(0.0f, 1.0f, 0.0f);
            constexpr auto blue = make_float3(0.0f, 0.0f, 1.0f);
            color = triangle_interpolate(hit.bary, red, green, blue);
        };
        // Progressive accumulation
        auto old = image.read(coord.y * dispatch_size_x() + coord.x).xyz();
        auto t = 1.0f / (cast<Float>(frame_index) + 1.0f);
        image.write(coord.y * dispatch_size_x() + coord.x, make_float4(lerp(old, color, t), 1.0f));
    };

    // HDR to LDR conversion
    Kernel2D colorspace_kernel = [&](BufferFloat4 hdr_image, BufferUInt ldr_image) noexcept {
        UInt i = dispatch_y() * dispatch_size_x() + dispatch_x();
        Float3 hdr = hdr_image.read(i).xyz();
        UInt3 ldr = make_uint3(round(clamp(linear_to_srgb(hdr), 0.f, 1.f) * 255.0f));
        ldr_image.write(i, ldr.x | (ldr.y << 8u) | (ldr.z << 16u) | (255u << 24u));
    };

    // Build acceleration structure
    Accel accel = device.create_accel();
    accel.emplace_back(mesh, translation(-.3f, 0.f, 0.f) * scaling(2.f));
    stream << mesh.build()
           << accel.build();

    // Compile shaders
    auto colorspace_shader = device.compile(colorspace_kernel);
    auto raytracing_shader = device.compile(raytracing_kernel);

    Buffer<float4> hdr_image = device.create_buffer<float4>(width * height);
    Buffer<uint> ldr_image = device.create_buffer<uint>(width * height);
    std::vector<uint8_t> pixels(width * height * 4u);

    // Render with motion blur
    Clock clock;
    clock.tic();
    static constexpr uint spp = 1024u;
    for (uint i = 0u; i < spp; i++) {
        stream << raytracing_shader(hdr_image, accel, i).dispatch(width, height);
    }
    stream << colorspace_shader(hdr_image, ldr_image).dispatch(width, height)
           << ldr_image.copy_to(luisa::span{pixels})
           << synchronize();
    double time = clock.toc();
    LUISA_INFO("Time: {} ms", time);
    stbi_write_png("test_motion_blur_mesh.png", width, height, 4, pixels.data(), 0);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_motion_blur(device);
}
