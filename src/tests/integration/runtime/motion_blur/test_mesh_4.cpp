// Motion Blur Ray Tracing Test - Motion Instance SRT Transform (4 quadrants)
// Tests all four SRT components of instance-level motion blur, one per quadrant:
//
//   Top-left:     Rotation   — Y-axis rotation from 0 to 45 degrees
//   Top-right:    Translation — slides right by 0.5 units
//   Bottom-left:  Scale      — uniform scale from 1.0 to 1.8
//   Bottom-right: Shear      — shear (xy) from 0 to 0.5
//
// Each quadrant contains a static triangle mesh wrapped in a MotionInstance
// with SRT keyframes exercising one specific SRT component.
// The triangles are colored by instance_user_id so each quadrant is visually
// distinguishable: red, green, blue, yellow.

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

void test_motion_blur_mesh_4(Device &device) {

    log_level_verbose();

    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);

    static constexpr uint width = 512u;
    static constexpr uint height = 512u;
    static constexpr uint instance_count = 4u;

    auto stream = device.create_stream(StreamTag::GRAPHICS);

    // Camera at (0,0,3) with 60-degree FOV gives visible range ~[-1.73, 1.73] at z=0.
    // Quadrant centers at (+/-0.87, +/-0.87). Use a small triangle (radius ~0.3)
    // so each fits comfortably within its quadrant.
    std::array vertices{
        float3(-0.3f, -0.3f, 0.0f),
        float3(0.3f, -0.3f, 0.0f),
        float3(0.0f, 0.3f, 0.0f),
    };
    std::array indices{0u, 1u, 2u};

    auto vertex_buffer = device.create_buffer<float3>(3u);
    auto triangle_buffer = device.create_buffer<Triangle>(1u);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{indices});

    // All four instances share the same static mesh
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);

    // Common motion option: SRT mode, 2 keyframes, time [0, 1]
    auto make_motion_option = []() {
        AccelMotionOption opt;
        opt.mode = AccelMotionMode::SRT;
        opt.keyframe_count = 2u;
        opt.time_start = 0.f;
        opt.time_end = 1.f;
        return opt;
    };

    // Quadrant center offsets (camera looks along -Z, so screen Y = world Y)
    //   index 0: top-left     (-0.87, +0.87)  — Rotation
    //   index 1: top-right    (+0.87, +0.87)  — Translation
    //   index 2: bottom-left  (-0.87, -0.87)  — Scale
    //   index 3: bottom-right (+0.87, -0.87)  — Shear
    static constexpr float qx[4] = {-0.87f, 0.87f, -0.87f, 0.87f};
    static constexpr float qy[4] = {0.87f, 0.87f, -0.87f, -0.87f};

    // Helper: identity SRT at a given quadrant center
    auto make_identity_srt = [](float tx, float ty) {
        return MotionInstanceTransformSRT{
            .pivot = {0.f, 0.f, 0.f},
            .quaternion = {0.f, 0.f, 0.f, 1.f},
            .scale = {1.f, 1.f, 1.f},
            .shear = {0.f, 0.f, 0.f},
            .translation = {tx, ty, 0.f}};
    };

    // --- Instance 0: Rotation (top-left) ---
    // Rotates around Y axis from 0 to 45 degrees
    auto mi0 = device.create_motion_instance(mesh, make_motion_option());
    {
        auto angle0 = radians(0.f);
        auto angle1 = radians(45.f);
        MotionInstanceTransformSRT k0{
            .pivot = {0.f, 0.f, 0.f},
            .quaternion = {0.f, sin(angle0 / 2.f), 0.f, cos(angle0 / 2.f)},
            .scale = {1.f, 1.f, 1.f},
            .shear = {0.f, 0.f, 0.f},
            .translation = {qx[0], qy[0], 0.f}};
        MotionInstanceTransformSRT k1{
            .pivot = {0.f, 0.f, 0.f},
            .quaternion = {0.f, sin(angle1 / 2.f), 0.f, cos(angle1 / 2.f)},
            .scale = {1.f, 1.f, 1.f},
            .shear = {0.f, 0.f, 0.f},
            .translation = {qx[0], qy[0], 0.f}};
        luisa::vector<MotionInstanceTransformSRT> kfs{k0, k1};
        mi0.set_keyframes(kfs);
    }

    // --- Instance 1: Translation (top-right) ---
    // Slides right by 0.5 units
    auto mi1 = device.create_motion_instance(mesh, make_motion_option());
    {
        MotionInstanceTransformSRT k0 = make_identity_srt(qx[1] - 0.25f, qy[1]);
        MotionInstanceTransformSRT k1 = make_identity_srt(qx[1] + 0.25f, qy[1]);
        luisa::vector<MotionInstanceTransformSRT> kfs{k0, k1};
        mi1.set_keyframes(kfs);
    }

    // --- Instance 2: Scale (bottom-left) ---
    // Uniform scale from 1.0 to 1.8
    auto mi2 = device.create_motion_instance(mesh, make_motion_option());
    {
        MotionInstanceTransformSRT k0 = make_identity_srt(qx[2], qy[2]);
        k0.scale[0] = 1.0f;
        k0.scale[1] = 1.0f;
        k0.scale[2] = 1.0f;
        MotionInstanceTransformSRT k1 = make_identity_srt(qx[2], qy[2]);
        k1.scale[0] = 1.8f;
        k1.scale[1] = 1.8f;
        k1.scale[2] = 1.8f;
        luisa::vector<MotionInstanceTransformSRT> kfs{k0, k1};
        mi2.set_keyframes(kfs);
    }

    // --- Instance 3: Shear (bottom-right) ---
    // Shear xy from 0 to 0.5
    auto mi3 = device.create_motion_instance(mesh, make_motion_option());
    {
        MotionInstanceTransformSRT k0 = make_identity_srt(qx[3], qy[3]);
        // shear[0] = a (xy shear), shear[1] = b (xz shear), shear[2] = c (yz shear)
        k0.shear[0] = 0.f;
        MotionInstanceTransformSRT k1 = make_identity_srt(qx[3], qy[3]);
        k1.shear[0] = 0.5f;
        luisa::vector<MotionInstanceTransformSRT> kfs{k0, k1};
        mi3.set_keyframes(kfs);
    }

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

    // Camera ray generation - looking at origin from front
    Callable generate_ray = [](Float2 p) noexcept {
        constexpr auto origin = make_float3(0.f, 0.f, 3.f);
        constexpr auto target = make_float3(0.f, 0.f, 0.f);
        auto up = make_float3(0.f, 1.f, 0.f);
        auto front = normalize(target - origin);
        auto right = normalize(cross(front, up));
        up = cross(right, front);
        auto fov = radians(60.f);
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
    Kernel2D raytracing_kernel = [&](BufferFloat4 image, AccelVar accel, UInt frame_index) noexcept {
        auto coord = dispatch_id().xy();
        auto color = def<float3>(0.1f, 0.1f, 0.15f);
        auto u = rand(frame_index, coord);
        auto ray = generate_ray(make_float2(coord) + u.xy());
        auto time = u.z * 1.f;
        auto hit = accel.intersect_motion(ray, time, {});
        $if (hit->is_triangle()) {
            // Per-instance color: red, green, blue, yellow
            auto inst = accel.instance_user_id(hit.inst);
            auto c0 = make_float3(1.0f, 0.2f, 0.2f);// red   — rotation
            auto c1 = make_float3(0.2f, 1.0f, 0.2f);// green — translation
            auto c2 = make_float3(0.2f, 0.4f, 1.0f);// blue  — scale
            auto c3 = make_float3(1.0f, 0.9f, 0.2f);// yellow — shear
            // Modulate by barycentric coordinates for depth
            auto bary_brightness = 0.5f + 0.5f * (hit.bary.x + hit.bary.y);
            $if (inst == 0u) { color = c0 * bary_brightness; };
            $if (inst == 1u) { color = c1 * bary_brightness; };
            $if (inst == 2u) { color = c2 * bary_brightness; };
            $if (inst == 3u) { color = c3 * bary_brightness; };
        };
        // Progressive accumulation
        UInt pixel_index = coord.y * dispatch_size_x() + coord.x;
        $if (frame_index == 0u) {
            image.write(pixel_index, make_float4(color, 1.0f));
        }
        $else {
            auto old = image.read(pixel_index).xyz();
            auto t = 1.0f / (cast<Float>(frame_index) + 1.0f);
            image.write(pixel_index, make_float4(lerp(old, color, t), 1.0f));
        };
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
    accel.emplace_back(mi0, make_float4x4(1.f), 0xffu, true, 0u);
    accel.emplace_back(mi1, make_float4x4(1.f), 0xffu, true, 1u);
    accel.emplace_back(mi2, make_float4x4(1.f), 0xffu, true, 2u);
    accel.emplace_back(mi3, make_float4x4(1.f), 0xffu, true, 3u);
    stream << mesh.build()
           << mi0.build()
           << mi1.build()
           << mi2.build()
           << mi3.build()
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
    auto output_directory = std::filesystem::path{opts.output_dir};
    std::error_code output_error;
    std::filesystem::create_directories(output_directory, output_error);
    boost::ut::expect(!output_error)
        << luisa::format("Failed to create output directory '{}': {}",
                         output_directory.string(), output_error.message());
    auto output_path = output_directory / "test_motion_blur_mesh_4.png";
    auto saved = !output_error &&
                 stbi_write_png(output_path.string().c_str(), width, height, 4,
                                pixels.data(), static_cast<int>(width * 4u)) != 0;
    boost::ut::expect(saved)
        << luisa::format("Failed to save output image '{}'.", output_path.string());
    if (!saved) { return; }
    LUISA_INFO("Saved output to {}", output_path.string());
    if (opts.compare_path) {
        auto result = luisa::test::compare_with_reference_file(
            pixels.data(), static_cast<int>(width), static_cast<int>(height), 4,
            *opts.compare_path);
        LUISA_INFO("Reference comparison [test_motion_blur_mesh_4]: {} ({})",
                   result.passed ? "PASSED" : "FAILED", result.message);
        boost::ut::expect(result.passed) << result.message;
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_motion_blur_mesh_4(device);
}
