// Minimal reproduction of Smaray MegaKernel crash on VK backend with motion blur.
//
// Root cause: When a kernel uses BOTH intersect() (inline ray query) AND
// intersect_motion() (pipeline TraceRay), the VK backend compiles it as a
// lib_6_x RT pipeline shader. The generated HLSL mixes RayQuery::TraceRayInline
// (from _TraceClosest) with TraceRay (from _TraceClosestMotion) in the same
// library shader. DXC fails to compile this combination to SPIR-V because
// SPV_KHR_ray_query is not enabled, and even if it were, mixing inline and
// pipeline ray tracing in one shader is problematic on some drivers.
//
// This test reproduces the crash by using both accel.intersect() and
// accel.intersect_motion() in the same kernel — exactly what Smaray's
// MegaKernel does (TraceTemporal for motion blur + Trace for shadow rays).

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

void test_motion_blur_mixed_trace(Device &device) {

    log_level_verbose();

    static constexpr uint width = 256u;
    static constexpr uint height = 256u;
    static constexpr auto mesh_keyframe_count = 2u;

    auto stream = device.create_stream(StreamTag::GRAPHICS);

    // Simple triangle with 2 keyframes (vertex motion)
    std::array vertices{
        // keyframe 0
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(-0.1f, 0.5f, 0.0f),
        // keyframe 1
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.1f, 0.5f, 0.0f),
    };
    std::array indices{0u, 1u, 2u};

    auto vertex_buffer = device.create_buffer<float3>(3u * mesh_keyframe_count);
    auto triangle_buffer = device.create_buffer<Triangle>(1u);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{indices});

    // Mesh with motion blur
    AccelOption mesh_option;
    mesh_option.motion.keyframe_count = mesh_keyframe_count;
    mesh_option.motion.time_start = 0.f;
    mesh_option.motion.time_end = 1.f;
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer, mesh_option);

    // Build acceleration structure
    Accel accel = device.create_accel();
    accel.emplace_back(mesh, scaling(2.f));
    stream << mesh.build()
           << accel.build();

    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                      12.92f * x,
                      x <= 0.00031308f);
    };
    // HDR to LDR conversion
    Kernel2D colorspace_kernel = [&](BufferFloat4 hdr_image, BufferUInt ldr_image) noexcept {
        UInt i = dispatch_y() * dispatch_size_x() + dispatch_x();
        Float3 hdr = hdr_image.read(i).xyz();
        UInt3 ldr = make_uint3(round(clamp(linear_to_srgb(hdr), 0.f, 1.f) * 255.0f));
        ldr_image.write(i, ldr.x | (ldr.y << 8u) | (ldr.z << 16u) | (255u << 24u));
    };
    auto colorspace_shader = device.compile(colorspace_kernel);
    // === THE KEY KERNEL ===
    // This kernel uses intersect_motion() + intersect() + traverse() (query_all)
    // which is exactly what Smaray's MegaKernel does:
    //   - intersect_motion() for primary rays (motion blur sampling)
    //   - intersect() for shadow rays
    //   - traverse() for ray query traversal (e.g., alpha-tested geometry)
    //
    // The combination trace_closest=1 + trace_closest_mb=1 + query_all=1
    // triggers the crash on VK backend.
    Kernel2D mixed_trace_kernel = [&](BufferFloat4 image, AccelVar accel_arg, UInt frame_index) noexcept {
        auto coord = dispatch_id().xy();
        auto size = dispatch_size().xy();
        auto uv = (make_float2(coord) + 0.5f) / make_float2(size) * 2.f - 1.f;

        // Generate primary ray
        auto ray_origin = make_float3(0.f, 0.f, 3.f);
        auto ray_dir = normalize(make_float3(uv.x, uv.y, -1.f));
        auto ray = make_ray(ray_origin, ray_dir);

        // Random time per pixel per frame for motion blur
        auto seed = coord.x * 73856093u ^ coord.y * 19349663u ^ frame_index * 83492791u;
        auto time = cast<Float>(seed % 65536u) / 65536.f;

        auto color = def(make_float3(0.f));

        // PRIMARY RAY: use intersect_motion (motion blur)
        // This causes requires_motion_blur=1, trace_closest_mb=1
        auto hit = accel_arg.intersect_motion(ray, time, {});

        $if (hit->is_triangle()) {
            color = make_float3(hit.bary, 0.f);

            // SHADOW RAY: use plain intersect
            // This causes trace_closest=1
            auto shadow_origin = ray_origin + ray_dir * hit->committed_ray_t;
            auto shadow_dir = normalize(make_float3(1.f, 1.f, 1.f));
            auto shadow_ray = make_ray(shadow_origin, shadow_dir, 0.001f, 100.f);
            auto shadow_hit = accel_arg.intersect(shadow_ray, {});
            $if (shadow_hit->is_triangle()) {
                color *= 0.3f;
            };

            // RAY QUERY TRAVERSAL: use traverse() (query_all)
            // This causes query_all=1, which uses RayQuery::TraceRayInline
            // in the generated HLSL. Combined with TraceRay from motion blur,
            // this may crash DXC SPIR-V compilation.
            auto query_ray = make_ray(shadow_origin, -shadow_dir, 0.001f, 100.f);
            Var<CommittedHit> query_hit = accel_arg.traverse(query_ray, {})
                .on_surface_candidate([&](auto &c) noexcept {
                    c.commit();
                })
                .trace();
            $if (!query_hit->miss()) {
                color *= 0.7f;
            };
        };

        // Progressive accumulation: blend with previous frames
        auto idx = coord.y * size.x + coord.x;
        auto old = image.read(idx).xyz();
        auto t = 1.0f / (cast<Float>(frame_index) + 1.0f);
        image.write(idx, make_float4(lerp(old, color, t), 1.f));
    };

    // Compile — this is where the crash happens on VK backend
    auto shader = device.compile(mixed_trace_kernel);

    // If we get here, compilation succeeded. Run it.
    Buffer<float4> hdr_image = device.create_buffer<float4>(width * height);
    Buffer<uint> ldr_image = device.create_buffer<uint>(width * height);
    std::vector<uint8_t> pixels(width * height * 4u);
    static constexpr uint spp = 1024u;
    for (uint i = 0u; i < spp; i++) {
        stream << shader(hdr_image, accel, i).dispatch(width, height);
    }
    stream << colorspace_shader(hdr_image, ldr_image).dispatch(width, height)
           << ldr_image.copy_to(luisa::span{pixels})
           << synchronize();
    stbi_write_png("test_motion_blur_mesh_5.png", width, height, 4, pixels.data(), 0);

    LUISA_INFO("test_motion_blur_mixed_trace passed (no crash)");
}

static inline const auto reg = [] {
    "test_motion_blur_mixed_trace"_test = [] {
        auto dc = luisa::test::create_device_from_ut();
        if (!dc) return;
        auto &device = dc->device;
        test_motion_blur_mixed_trace(device);
    };
    return 0;
}();

int main() {}
