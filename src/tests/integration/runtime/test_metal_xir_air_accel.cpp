// Focused strict-AIR test for Metal acceleration-structure lowering.
// This test covers:
// - closest-hit and any-hit triangle intersection
// - ray visibility-mask filtering
// - instance transform, user ID, and visibility-mask queries

#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <cmath>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto instance_visibility = static_cast<uint8_t>(0x5au);
constexpr auto instance_user_id = 0x12345678u;

void test_metal_xir_air_accel(Device &device) {
    log_level_verbose();

    constexpr std::array vertices{
        float3{-1.0f, -1.0f, 0.0f},
        float3{1.0f, -1.0f, 0.0f},
        float3{0.0f, 1.0f, 0.0f}};
    constexpr std::array triangles{Triangle{0u, 1u, 2u}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f),
                       instance_visibility, true, instance_user_id);

    auto result_buffer = device.create_buffer<uint>(8u);
    auto hit_data_buffer = device.create_buffer<float4>(1u);
    auto transform_buffer = device.create_buffer<float4x4>(1u);

    Kernel1D trace_kernel = [](AccelVar accel,
                               BufferUInt results,
                               BufferFloat4 hit_data,
                               BufferVar<float4x4> transforms) noexcept {
        auto hit_ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                                make_float3(0.0f, 0.0f, -1.0f),
                                0.0f, 10.0f);
        auto miss_ray = make_ray(make_float3(2.0f, 2.0f, 1.0f),
                                 make_float3(0.0f, 0.0f, -1.0f),
                                 0.0f, 10.0f);

        Var<TriangleHit> closest = accel.intersect(
            hit_ray, {.visibility_mask = 0x08u});
        Bool any_visible = accel.intersect_any(
            hit_ray, {.visibility_mask = 0x08u});
        Bool any_filtered = accel.intersect_any(
            hit_ray, {.visibility_mask = 0x01u});
        Var<TriangleHit> closest_miss = accel.intersect(
            miss_ray, {.visibility_mask = 0x08u});

        results.write(0u, ite(closest->miss(), 0u, 1u));
        results.write(1u, closest.inst);
        results.write(2u, closest.prim);
        results.write(3u, ite(any_visible, 1u, 0u));
        results.write(4u, ite(any_filtered, 1u, 0u));
        results.write(5u, ite(closest_miss->miss(), 1u, 0u));
        results.write(6u, accel.instance_user_id(0u));
        results.write(7u, accel.instance_visibility_mask(0u));
        hit_data.write(0u, make_float4(closest.bary, closest->distance(), 0.0f));
        transforms.write(0u, accel.instance_transform(0u));
    };
    auto trace_shader = device.compile(trace_kernel);

    std::array<uint, 8u> results{};
    std::array<float4, 1u> hit_data{};
    std::array<float4x4, 1u> transforms{};

    auto stream = device.create_stream();
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << trace_shader(accel, result_buffer, hit_data_buffer,
                           transform_buffer)
                  .dispatch(1u)
           << result_buffer.copy_to(luisa::span{results})
           << hit_data_buffer.copy_to(luisa::span{hit_data})
           << transform_buffer.copy_to(luisa::span{transforms})
           << synchronize();

    expect(results[0] == 1u) << "closest trace should hit the triangle";
    expect(results[1] == 0u) << "closest trace should return instance zero";
    expect(results[2] == 0u) << "closest trace should return primitive zero";
    expect(results[3] == 1u) << "any-hit trace should see a visible instance";
    expect(results[4] == 0u) << "non-overlapping ray mask should filter the instance";
    expect(results[5] == 1u) << "ray outside the triangle should miss";
    expect(results[6] == instance_user_id) << "instance user ID should round-trip";
    expect(results[7] == instance_visibility) << "instance visibility should round-trip";

    constexpr auto epsilon = 1.0e-5f;
    expect(std::abs(hit_data[0].x - 0.25f) < epsilon)
        << "unexpected first barycentric coordinate";
    expect(std::abs(hit_data[0].y - 0.5f) < epsilon)
        << "unexpected second barycentric coordinate";
    expect(std::abs(hit_data[0].z - 1.0f) < epsilon)
        << "unexpected hit distance";

    for (auto column = 0u; column < 4u; column++) {
        for (auto row = 0u; row < 4u; row++) {
            auto expected = column == row ? 1.0f : 0.0f;
            expect(std::abs(transforms[0][column][row] - expected) < epsilon)
                << "instance transform should be identity";
        }
    }
}

void test_metal_xir_air_primitive_motion(Device &device) {
    constexpr std::array vertices{
        // keyframe 0: z = 0
        float3{-1.0f, -1.0f, 0.0f},
        float3{1.0f, -1.0f, 0.0f},
        float3{0.0f, 1.0f, 0.0f},
        // keyframe 1: z = -1
        float3{-1.0f, -1.0f, -1.0f},
        float3{1.0f, -1.0f, -1.0f},
        float3{0.0f, 1.0f, -1.0f}};
    constexpr std::array triangles{Triangle{0u, 1u, 2u}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    AccelOption motion_option{};
    motion_option.motion.keyframe_count = 2u;
    motion_option.motion.time_start = 0.0f;
    motion_option.motion.time_end = 1.0f;
    auto mesh = device.create_mesh(
        vertex_buffer, triangle_buffer, motion_option);
    auto accel = device.create_accel();
    accel.emplace_back(mesh);
    auto results_buffer = device.create_buffer<float4>(1u);

    Kernel1D kernel = [](AccelVar accel,
                         BufferFloat4 results) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 10.0f);
        auto at_start = accel.intersect_motion(ray, 0.0f, {});
        auto at_end = accel.intersect_motion(ray, 1.0f, {});
        auto any_mid = accel.intersect_any_motion(ray, 0.5f, {});
        results.write(
            0u, make_float4(
                    at_start->distance(), at_end->distance(),
                    ite(any_mid, 1.0f, 0.0f),
                    ite(at_end->is_triangle(), 1.0f, 0.0f)));
    };
    auto shader = device.compile(kernel);

    std::array<float4, 1u> results{};
    auto stream = device.create_stream();
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << shader(accel, results_buffer).dispatch(1u)
           << results_buffer.copy_to(luisa::span{results})
           << synchronize();

    constexpr auto epsilon = 1.0e-4f;
    expect(std::abs(results[0].x - 1.0f) < epsilon)
        << "primitive-motion trace at shutter start has wrong distance";
    expect(std::abs(results[0].y - 2.0f) < epsilon)
        << "primitive-motion trace at shutter end has wrong distance";
    expect(std::abs(results[0].z - 1.0f) < epsilon)
        << "primitive-motion any-hit should report the moving triangle";
    expect(std::abs(results[0].w - 1.0f) < epsilon)
        << "primitive-motion closest hit should retain triangle semantics";
}

void test_metal_xir_air_motion_instance_matrix(Device &device) {
    constexpr std::array vertices{
        float3{-1.0f, -1.0f, 0.0f},
        float3{1.0f, -1.0f, 0.0f},
        float3{0.0f, 1.0f, 0.0f}};
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
    auto make_translation = [](float x, float z) noexcept {
        auto matrix = make_float4x4(1.0f);
        matrix[3u].x = x;
        matrix[3u].z = z;
        return matrix;
    };
    std::array keyframes{
        make_translation(0.0f, 0.0f),
        make_translation(1.0f, -1.0f)};
    motion_instance.set_keyframes(luisa::span{keyframes});

    AccelOption accel_option{};
    accel_option.allow_update = true;
    auto accel = device.create_accel(accel_option);
    auto outer_transform =
        make_translation(0.0f, -0.25f) *
        rotation(make_float3(0.0f, 0.0f, 1.0f), radians(90.0f));
    accel.emplace_back(
        motion_instance, outer_transform,
        0xffu, true, 41u);
    accel.emplace_back(
        mesh, make_translation(3.0f, -0.5f),
        0xffu, true, 42u);

    auto distances_buffer = device.create_buffer<float4>(1u);
    auto metadata_buffer = device.create_buffer<uint4>(1u);
    Kernel1D kernel = [](AccelVar accel, BufferFloat4 distances,
                         BufferUInt4 metadata) noexcept {
        auto moving_start_ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
        auto moving_mid_ray = make_ray(
            make_float3(0.0f, 0.5f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
        auto moving_end_ray = make_ray(
            make_float3(0.0f, 1.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
        auto static_ray = make_ray(
            make_float3(3.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
        auto moving_start =
            accel.intersect_motion(moving_start_ray, 0.0f, {});
        auto moving_end =
            accel.intersect_motion(moving_end_ray, 1.0f, {});
        auto static_mid = accel.intersect_motion(static_ray, 0.5f, {});
        distances.write(
            0u, make_float4(
                    moving_start->distance(), moving_end->distance(),
                    static_mid->distance(),
                    ite(accel.intersect_any_motion(
                            moving_mid_ray, 0.5f, {}),
                        1.0f, 0.0f)));
        metadata.write(
            0u, make_uint4(
                    moving_start.inst, moving_end.inst, static_mid.inst,
                    accel.instance_user_id(static_mid.inst)));
    };
    auto shader = device.compile(kernel);

    std::array<float4, 1u> distances{};
    std::array<uint4, 1u> metadata{};
    auto stream = device.create_stream();
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << motion_instance.build()
           << accel.build()
           << shader(accel, distances_buffer, metadata_buffer).dispatch(1u)
           << distances_buffer.copy_to(luisa::span{distances})
           << metadata_buffer.copy_to(luisa::span{metadata})
           << synchronize();

    constexpr auto epsilon = 1.0e-4f;
    expect(std::abs(distances[0].x - 1.25f) < epsilon)
        << "matrix MotionInstance start transform was not composed";
    expect(std::abs(distances[0].y - 2.25f) < epsilon)
        << "matrix MotionInstance end transform was not composed";
    expect(std::abs(distances[0].z - 1.5f) < epsilon)
        << "static instance in a motion TLAS has the wrong transform";
    expect(std::abs(distances[0].w - 1.0f) < epsilon)
        << "matrix MotionInstance any-hit failed";
    expect(metadata[0].x == 0u && metadata[0].y == 0u)
        << "matrix MotionInstance returned the wrong instance index";
    expect(metadata[0].z == 1u && metadata[0].w == 42u)
        << "static instance metadata was not preserved in a motion TLAS";

    keyframes = {
        make_translation(0.0f, -2.0f),
        make_translation(1.0f, -3.0f)};
    motion_instance.set_keyframes(luisa::span{keyframes});
    stream << motion_instance.build()
           << accel.build()
           << shader(accel, distances_buffer, metadata_buffer).dispatch(1u)
           << distances_buffer.copy_to(luisa::span{distances})
           << synchronize();
    expect(std::abs(distances[0].x - 3.25f) < epsilon)
        << "matrix MotionInstance refit did not import rebuilt keyframes";
    expect(std::abs(distances[0].y - 4.25f) < epsilon)
        << "matrix MotionInstance refit has the wrong end keyframe";
}

void test_metal_xir_air_curve_motion(Device &device) {
    constexpr auto basis = CurveBasis::PIECEWISE_LINEAR;
    constexpr std::array control_points{
        // keyframe 0: z = 0
        float4{-1.0f, 0.0f, 0.0f, 0.2f},
        float4{1.0f, 0.0f, 0.0f, 0.2f},
        // keyframe 1: z = -1
        float4{-1.0f, 0.0f, -1.0f, 0.2f},
        float4{1.0f, 0.0f, -1.0f, 0.2f}};
    constexpr std::array segments{0u};

    auto control_point_buffer =
        device.create_buffer<float4>(control_points.size());
    auto segment_buffer = device.create_buffer<uint>(segments.size());
    AccelOption motion_option{};
    motion_option.motion.keyframe_count = 2u;
    motion_option.motion.time_start = 0.0f;
    motion_option.motion.time_end = 1.0f;
    auto curve = device.create_curve(
        basis, control_point_buffer, segment_buffer, motion_option);
    auto accel = device.create_accel();
    accel.emplace_back(curve);
    auto results_buffer = device.create_buffer<float4>(1u);

    Kernel1D kernel = [](AccelVar accel,
                         BufferFloat4 results) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 10.0f);
        auto at_start = accel.intersect_motion(
            ray, 0.0f, {.curve_bases = {basis}});
        auto at_end = accel.intersect_motion(
            ray, 1.0f, {.curve_bases = {basis}});
        results.write(
            0u, make_float4(
                    at_start->distance(), at_end->distance(),
                    at_start->curve_parameter(),
                    ite(at_end->is_curve(), 1.0f, 0.0f)));
    };
    auto shader = device.compile(kernel);

    std::array<float4, 1u> results{};
    auto stream = device.create_stream();
    stream << control_point_buffer.copy_from(
                  luisa::span{control_points})
           << segment_buffer.copy_from(luisa::span{segments})
           << curve.build()
           << accel.build()
           << shader(accel, results_buffer).dispatch(1u)
           << results_buffer.copy_to(luisa::span{results})
           << synchronize();

    constexpr auto epsilon = 1.0e-4f;
    expect(std::abs(results[0].x - 0.8f) < epsilon)
        << "curve-motion trace at shutter start has wrong distance";
    expect(std::abs(results[0].y - 1.8f) < epsilon)
        << "curve-motion trace at shutter end has wrong distance";
    expect(std::abs(results[0].z - 0.5f) < epsilon)
        << "curve-motion trace has wrong curve parameter";
    expect(std::abs(results[0].w - 1.0f) < epsilon)
        << "curve-motion closest hit should retain curve semantics";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_metal_xir_air_accel(dc->device);
    test_metal_xir_air_primitive_motion(dc->device);
    test_metal_xir_air_motion_instance_matrix(dc->device);
    test_metal_xir_air_curve_motion(dc->device);
}
