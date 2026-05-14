// Motion Blur Ray Tracing Test - GPU-side set_instance_motion_matrix
// This test reproduces a Vulkan backend crash when the shader kernel calls
// accel.set_instance_motion_matrix() to update motion instance transforms
// from the GPU side (as Smaray's MegaKernel does), combined with
// accel.intersect_motion() for ray tracing.
//
// The crash occurs because RAY_TRACING_SET_INSTANCE_MOTION_MATRIX (CallOp 276)
// is not handled in the HLSL codegen's function_codegen.cpp switch statement.

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

void test_motion_blur_mesh_7(Device &device) {

    log_level_verbose();

    static constexpr uint width = 256u;
    static constexpr uint height = 256u;

    auto stream = device.create_stream(StreamTag::GRAPHICS);

    // Simple triangle
    std::array vertices{
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.0f, 0.5f, 0.0f),
    };
    std::array indices{0u, 1u, 2u};

    auto vertex_buffer = device.create_buffer<float3>(3u);
    auto triangle_buffer = device.create_buffer<Triangle>(1u);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{indices});

    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);

    // Create a motion instance with Matrix mode, 2 keyframes
    AccelMotionOption motion_opt;
    motion_opt.mode = AccelMotionMode::MATRIX;
    motion_opt.keyframe_count = 2u;
    motion_opt.time_start = 0.f;
    motion_opt.time_end = 1.f;

    auto mi = device.create_motion_instance(mesh, motion_opt);

    // Set initial keyframes on host side (identity for both)
    luisa::vector<MotionInstanceTransformMatrix> kfs(2u);
    kfs[0] = make_float4x4(1.f);
    kfs[1] = make_float4x4(1.f);
    mi.set_keyframes(kfs);

    // Build accel
    Accel accel = device.create_accel();
    accel.emplace_back(mi, make_float4x4(1.f), 0xffu, true, 0u);
    stream << mesh.build()
           << mi.build()
           << accel.build();

    // --- Kernel 1: Update motion instance transforms from GPU ---
    // This kernel only writes to the accel (no tracing).
    // It compiles as a compute shader.
    Kernel1D update_kernel = [&](AccelVar accel_var) noexcept {
        // Set keyframe 0: translate left
        auto m0 = make_float4x4(
            make_float4(1.f, 0.f, 0.f, 0.f),
            make_float4(0.f, 1.f, 0.f, 0.f),
            make_float4(0.f, 0.f, 1.f, 0.f),
            make_float4(-0.3f, 0.f, 0.f, 1.f));
        accel_var.set_instance_motion_matrix(0u, 0u, m0);

        // Set keyframe 1: translate right
        auto m1 = make_float4x4(
            make_float4(1.f, 0.f, 0.f, 0.f),
            make_float4(0.f, 1.f, 0.f, 0.f),
            make_float4(0.f, 0.f, 1.f, 0.f),
            make_float4(0.3f, 0.f, 0.f, 1.f));
        accel_var.set_instance_motion_matrix(0u, 1u, m1);
    };

    // --- Kernel 2: Ray trace with motion blur (read-only accel) ---
    Kernel2D trace_kernel = [&](BufferFloat4 image, AccelVar accel_var, UInt frame_index) noexcept {
        auto coord = dispatch_id().xy();
        auto uv = (make_float2(coord) + 0.5f) / make_float2(make_uint2(width, height)) * 2.f - 1.f;

        auto origin = make_float3(0.f, 0.f, 3.f);
        auto direction = normalize(make_float3(uv.x, -uv.y, -1.f));
        auto ray = make_ray(origin, direction);

        auto seed = coord.x + coord.y * width + frame_index * width * height;
        auto time = cast<Float>(seed % 1000u) / 1000.f;

        auto hit = accel_var.intersect_motion(ray, time, {});
        auto color = def<float3>(0.1f, 0.1f, 0.15f);
        $if (hit->is_triangle()) {
            color = make_float3(1.f, 0.5f + 0.5f * hit.bary.x, 0.2f);
        };

        auto old = image.read(coord.y * width + coord.x).xyz();
        auto t = 1.0f / (cast<Float>(frame_index) + 1.0f);
        image.write(coord.y * width + coord.x, make_float4(lerp(old, color, t), 1.0f));
    };

    // Compile shaders
    // update_kernel tests RAY_TRACING_SET_INSTANCE_MOTION_MATRIX codegen (compute shader path)
    // trace_kernel tests RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR codegen (RT pipeline path)
    auto update_shader = device.compile(update_kernel);
    auto trace_shader = device.compile(trace_kernel);

    Buffer<float4> hdr_image = device.create_buffer<float4>(width * height);
    std::vector<uint8_t> pixels(width * height * 4u);

    // Run update kernel then trace
    static constexpr uint spp = 64u;
    stream << update_shader(accel).dispatch(1u);
    stream << accel.build();
    for (uint i = 0u; i < spp; i++) {
        stream << trace_shader(hdr_image, accel, i).dispatch(width, height);
    }

    // Read back and save
    Buffer<uint> ldr_image = device.create_buffer<uint>(width * height);
    Kernel2D tonemap_kernel = [&](BufferFloat4 hdr, BufferUInt ldr) noexcept {
        UInt i = dispatch_y() * dispatch_size_x() + dispatch_x();
        Float3 c = clamp(hdr.read(i).xyz(), 0.f, 1.f);
        UInt3 rgb = make_uint3(round(c * 255.f));
        ldr.write(i, rgb.x | (rgb.y << 8u) | (rgb.z << 16u) | (255u << 24u));
    };
    auto tonemap_shader = device.compile(tonemap_kernel);
    stream << tonemap_shader(hdr_image, ldr_image).dispatch(width, height)
           << ldr_image.copy_to(luisa::span{pixels})
           << synchronize();

    stbi_write_png("test_motion_blur_mesh_7.png", width, height, 4, pixels.data(), 0);
    LUISA_INFO("test_motion_blur_mesh_7: rendered successfully (no crash)");
}

static inline const auto reg = [] {
    "test_motion_blur_mesh_7"_test = [] {
        auto dc = luisa::test::create_device_from_ut();
        if (!dc) return;
        auto &device = dc->device;
        test_motion_blur_mesh_7(device);
    };
    return 0;
}();

int main() {}
