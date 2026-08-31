// Executing image regression for Metal4 matrix MotionInstance traversal.
// Ray time follows the scanline, producing a visibly slanted moving triangle.

#include "ut/ut.hpp"
#include "reference_image.h"
#include "test_device.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto width = 160u;
constexpr auto height = 120u;

[[nodiscard]] float4x4 translation_matrix(float x, float y,
                                          float z) noexcept {
    auto matrix = make_float4x4(1.0f);
    matrix[3u] = make_float4(x, y, z, 1.0f);
    return matrix;
}

void render_motion_instance(Device &device,
                            const luisa::test::ImageTestOptions &opts) {
    auto matrix_motion = device.query("metal_motion_blur") == "true";
    auto component_motion =
        device.query("metal4_component_motion") == "true";
    auto address_driven =
        device.query("metal4_address_driven_acceleration_structures") ==
        "true";
    expect(matrix_motion)
        << "Metal4 test device does not support matrix motion blur";
    expect(component_motion == address_driven)
        << "Metal4 component-motion and address-driven guards disagree";
    if (!matrix_motion) { return; }

    constexpr std::array vertices{
        float3{-0.52f, -0.48f, 0.0f},
        float3{0.52f, -0.48f, 0.0f},
        float3{0.0f, 0.62f, 0.0f}};
    constexpr std::array triangles{Triangle{0u, 1u, 2u}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);

    AccelMotionOption motion_option{};
    motion_option.mode = AccelMotionMode::MATRIX;
    motion_option.keyframe_count = 2u;
    motion_option.time_start = 0.0f;
    motion_option.time_end = 1.0f;
    auto motion_instance =
        device.create_motion_instance(mesh, motion_option);
    const std::array keyframes{
        translation_matrix(-0.78f, 0.0f, 0.0f),
        translation_matrix(0.78f, 0.0f, -0.35f)};
    motion_instance.set_keyframes(luisa::span{keyframes});

    auto accel = device.create_accel();
    accel.emplace_back(
        motion_instance, translation_matrix(0.0f, 0.0f, -0.2f),
        0xffu, true, 73u);

    auto pixels_buffer = device.create_buffer<uint>(width * height);
    auto hits_buffer = device.create_buffer<uint>(width * height);
    Kernel2D render = [](AccelVar accel, BufferUInt pixels,
                         BufferUInt hits) noexcept {
        auto coord = dispatch_id().xy();
        auto pixel = make_float2(coord) + 0.5f;
        auto uv = pixel / make_float2(width, height);
        auto world = (uv * 2.0f - 1.0f) * make_float2(1.6f, -1.2f);
        auto ray = make_ray(
            make_float3(world, 1.5f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
        auto time = uv.y;
        auto hit = accel.intersect_motion(ray, time, {});
        auto did_hit = !hit->miss();
        auto background = lerp(
            make_float3(0.025f, 0.04f, 0.075f),
            make_float3(0.08f, 0.12f, 0.18f), uv.y);
        auto barycentric = make_float3(
            hit.bary.x, hit.bary.y,
            1.0f - hit.bary.x - hit.bary.y);
        auto foreground =
            0.18f + 0.82f * max(barycentric, make_float3(0.0f));
        auto color = select(background, foreground, did_hit);
        auto rgba = make_uint3(
            round(clamp(color, 0.0f, 1.0f) * 255.0f));
        auto index = coord.y * width + coord.x;
        pixels.write(index, rgba.x | (rgba.y << 8u) |
                                (rgba.z << 16u) | (255u << 24u));
        hits.write(index, ite(did_hit, 1u, 0u));
    };
    auto shader = device.compile(render);

    std::vector<uint> pixels(width * height);
    std::vector<uint> hits(width * height);
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << motion_instance.build()
           << accel.build()
           << shader(accel, pixels_buffer, hits_buffer)
                  .dispatch(width, height)
           << pixels_buffer.copy_to(luisa::span{pixels})
           << hits_buffer.copy_to(luisa::span{hits})
           << synchronize();

    auto hit_count = size_t{0u};
    auto upper_count = size_t{0u};
    auto lower_count = size_t{0u};
    auto upper_x_sum = uint64_t{0u};
    auto lower_x_sum = uint64_t{0u};
    auto min_x = width;
    auto max_x = 0u;
    for (auto y = 0u; y < height; y++) {
        for (auto x = 0u; x < width; x++) {
            if (hits[y * width + x] == 0u) { continue; }
            hit_count++;
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            if (y < height / 2u) {
                upper_count++;
                upper_x_sum += x;
            } else {
                lower_count++;
                lower_x_sum += x;
            }
        }
    }
    expect(hit_count > 500u)
        << "matrix MotionInstance render produced too few hit pixels";
    expect(upper_count > 100u && lower_count > 100u)
        << "matrix MotionInstance render did not cover both shutter halves";
    expect(max_x > min_x + width / 3u)
        << "matrix MotionInstance render has no visible motion extent";
    if (upper_count != 0u && lower_count != 0u) {
        auto upper_centroid = static_cast<double>(upper_x_sum) /
                              static_cast<double>(upper_count);
        auto lower_centroid = static_cast<double>(lower_x_sum) /
                              static_cast<double>(lower_count);
        LUISA_INFO("Metal4 matrix-motion centroids: upper {:.2f}, lower "
                   "{:.2f} pixels.",
                   upper_centroid, lower_centroid);
        expect(lower_centroid >
               upper_centroid + static_cast<double>(width) / 16.0)
            << "matrix MotionInstance render does not move with ray time";
    }

    auto output_directory = std::filesystem::path{opts.output_dir};
    std::error_code error;
    std::filesystem::create_directories(output_directory, error);
    expect(!error) << "failed to create motion-render output directory";
    auto output_path =
        output_directory / "test_metal4_motion_instance.png";
    auto saved = !error &&
                 stbi_write_png(
                     output_path.string().c_str(), width, height, 4,
                     pixels.data(), static_cast<int>(width * sizeof(uint))) !=
                     0;
    expect(saved) << "failed to save Metal4 matrix-motion render";
    if (saved) {
        LUISA_INFO("Saved Metal4 matrix-motion render to {} ({} hit pixels).",
                   output_path.string(), hit_count);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);
    render_motion_instance(dc->device, opts);
}
