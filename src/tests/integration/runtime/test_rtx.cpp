// RTX Ray Tracing Test
// This test demonstrates basic ray tracing using the RTX acceleration structure.
// It renders a simple animated triangle scene with progressive accumulation.
// Features demonstrated:
// - RTX acceleration structure (Accel) for ray-triangle intersection
// - Progressive rendering with temporal accumulation
// - Halton sequence for quasi-random sampling
// - TEA (Tiny Encryption Algorithm) for random number generation
// - Dynamic mesh updates and transforms

#include "ut/ut.hpp"
#include "test_device.h"
#include "reference_image.h"

#include <filesystem>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_rtx(Device &device) {

    log_level_info();

    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);

    // Define a simple triangle mesh (3 vertices)
    std::array vertices{
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.0f, 0.5f, 0.0f)};
    std::array indices{0u, 1u, 2u};

    // Convert linear RGB to sRGB color space
    // Uses the standard sRGB transfer function with piecewise linear approximation
    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                      12.92f * x,
                      x <= 0.00031308f);
    };

    // Halton sequence generator for quasi-random sampling
    // Produces low-discrepancy sequences for better sampling distribution
    // base b must be prime (typically 2, 3, 5, 7, ...)
    Callable halton = [](UInt i, UInt b) noexcept {
        Float f = def(1.0f);
        Float invB = 1.0f / b;
        Float r = def(0.0f);
        $while (i > 0u) {
            f = f * invB;
            r = r + f * (i % b);
            i = i / b;
        };
        return r;
    };

    // TEA (Tiny Encryption Algorithm) for random number generation
    // Generates pseudo-random numbers from two seed values
    // Performs 4 rounds of encryption for good randomness
    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    // Generate 2D random numbers using TEA + Halton sequence
    // Combines TEA for seed scrambling with Halton for distribution
    Callable rand = [&](UInt f, UInt2 p) noexcept {
        UInt i = tea(p.x, p.y) + f;
        Float rx = halton(i, 2u);
        Float ry = halton(i, 3u);
        return make_float2(rx, ry);
    };

    // Ray tracing kernel: casts rays and accumulates color progressively
    // Uses RTX acceleration structure for ray-triangle intersection
    Kernel2D raytracing_kernel = [&](BufferFloat4 image, AccelVar accel, UInt frame_index) noexcept {
        UInt2 coord = dispatch_id().xy();
        // Generate normalized pixel coordinates with jitter for anti-aliasing
        Float2 p = (make_float2(coord) + rand(frame_index, coord)) /
                       make_float2(dispatch_size().xy()) * 2.0f -
                   1.0f;
        // Default background color (sky blue)
        Float3 color = def<float3>(0.3f, 0.5f, 0.7f);
        // Create ray from camera position through pixel
        Var<Ray> ray = make_ray(
            make_float3(p * make_float2(1.0f, -1.0f), 1.0f),
            make_float3(0.0f, 0.0f, -1.0f));
        // Trace ray against acceleration structure
        Var<TriangleHit> hit = accel.intersect(ray, {});
        // If hit, interpolate vertex colors using barycentric coordinates
        $if (!hit->miss()) {
            constexpr float3 red = float3(1.0f, 0.0f, 0.0f);
            constexpr float3 green = float3(0.0f, 1.0f, 0.0f);
            constexpr float3 blue = float3(0.0f, 0.0f, 1.0f);
            color = triangle_interpolate(hit.bary, red, green, blue);
        };
        // Progressive accumulation: blend new sample with previous samples
        Float3 old = image.read(coord.y * dispatch_size_x() + coord.x).xyz();
        Float t = 1.0f / (frame_index + 1.0f);
        image.write(coord.y * dispatch_size_x() + coord.x, make_float4(lerp(old, color, t), 1.0f));
    };

    // Convert HDR image to LDR for display/output
    Kernel2D colorspace_kernel = [&](BufferFloat4 hdr_image, BufferUInt ldr_image) noexcept {
        UInt i = dispatch_y() * dispatch_size_x() + dispatch_x();
        Float3 hdr = hdr_image.read(i).xyz();
        UInt3 ldr = make_uint3(round(clamp(linear_to_srgb(hdr), 0.f, 1.f) * 255.0f));
        ldr_image.write(i, ldr.x | (ldr.y << 8u) | (ldr.z << 16u) | (255u << 24u));
    };

    // Update instance transform in acceleration structure
    Kernel1D set_transform_kernel = [&](AccelVar accel, Float4x4 matrix, UInt offset) noexcept {
        accel.set_instance_transform(dispatch_id().x + offset, matrix);
    };

    // Create stream and buffers
    Stream stream = device.create_stream();
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(3u);
    Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(1u);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{indices});

    // Build acceleration structure with two instances of the triangle
    Accel accel = device.create_accel();
    Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    accel.emplace_back(mesh, scaling(1.5f));
    accel.emplace_back(mesh, translation(float3(-0.25f, 0.0f, 0.1f)) *
                                 rotation(float3(0.0f, 0.0f, 1.0f), 0.5f));
    stream << mesh.build()
           << accel.build();

    // Compile shaders
    auto colorspace_shader = device.compile(colorspace_kernel);
    auto raytracing_shader = device.compile(raytracing_kernel);
    auto set_transform_shader = device.compile(set_transform_kernel, {.name = "set_transforms"});

    // Setup image buffers
    static constexpr uint width = 512u;
    static constexpr uint height = 512u;
    Buffer<float4> hdr_image = device.create_buffer<float4>(width * height);
    Buffer<uint> ldr_image = device.create_buffer<uint>(width * height);
    std::vector<uint8_t> pixels(width * height * 4u);

    // Render animation with progressive accumulation
    Clock clock;
    clock.tic();
    static constexpr uint spp = 1024u;
    for (uint i = 0u; i < spp; i++) {
        float t = static_cast<float>(i) * (1.0f / spp);
        // Animate triangle vertex
        vertices[2].y = 0.5f - 0.2f * t;
        // Compute transform matrix for second instance
        float4x4 m = translation(float3(-0.25f + t * 0.15f, 0.0f, 0.1f)) *
                     rotation(float3(0.0f, 0.0f, 1.0f), 0.5f + t * 0.5f);

        stream << vertex_buffer.copy_from(luisa::span{vertices})
               << set_transform_shader(accel, m, 1u).dispatch(1)
               << mesh.build()
               << accel.build()
               << raytracing_shader(hdr_image, accel, i).dispatch(width, height);
        // Add third instance at halfway point
        if (i == 511u) {
            float4x4 mm = translation(make_float3(0.0f, 0.0f, 0.3f)) *
                          rotation(make_float3(0.0f, 0.0f, 1.0f), radians(180.0f));
            accel.emplace_back(mesh, mm, true);
            stream << accel.update_instance_buffer();
        }
    }
    stream << colorspace_shader(hdr_image, ldr_image).dispatch(width, height)
           << ldr_image.copy_to(luisa::span{pixels})
           << synchronize();
    double time = clock.toc();
    LUISA_INFO("Time: {} ms", time);
    stbi_write_png("test_rtx.png", width, height, 4, pixels.data(), 0);
    auto ref_dir = luisa::test::find_reference_dir(std::filesystem::path{boost::ut::detail::cfg::largv[0]}.parent_path());
    auto result = luisa::test::save_and_compare(
        pixels.data(), static_cast<int>(width), static_cast<int>(height), 4,
        "test_rtx", opts.output_dir, ref_dir, opts.update_reference);
    LUISA_INFO("Reference comparison: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
    if (!result.passed) {
        LUISA_ERROR("Reference comparison failed for test_rtx: {}", result.message);
        boost::ut::expect(static_cast<bool>(result.passed)) << result.message;
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
    test_rtx(device);
}
