// Exact HIP coverage for device-side motion-instance keyframe access.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto tolerance = 2.0e-4f;

void expect_near(float actual, float expected, luisa::string_view label) {
    expect(std::abs(actual - expected) < tolerance)
        << luisa::format("{}: got {}, expected {}", label, actual, expected);
}

void expect_matrix(const float4x4 &actual, const float4x4 &expected,
                   luisa::string_view label) {
    for (auto column = 0u; column < 4u; column++) {
        for (auto row = 0u; row < 4u; row++) {
            expect_near(actual[column][row], expected[column][row],
                        luisa::format("{}[{}][{}]", label, column, row));
        }
    }
}

void expect_srt(const MotionInstanceTransformSRT &actual,
                const MotionInstanceTransformSRT &expected,
                luisa::string_view label) {
    auto check = [&](const float *lhs, const float *rhs,
                     size_t count, luisa::string_view field) {
        for (auto i = 0u; i < count; i++) {
            expect_near(lhs[i], rhs[i],
                        luisa::format("{}.{}[{}]", label, field, i));
        }
    };
    check(actual.pivot, expected.pivot, 3u, "pivot");
    check(actual.quaternion, expected.quaternion, 4u, "quaternion");
    check(actual.scale, expected.scale, 3u, "scale");
    check(actual.shear, expected.shear, 3u, "shear");
    check(actual.translation, expected.translation, 3u, "translation");
}

[[nodiscard]] MotionInstanceTransformSRT make_srt(
    float3 pivot, float3 scale, float3 shear, float3 translation) noexcept {
    return MotionInstanceTransformSRT{
        .pivot = {pivot.x, pivot.y, pivot.z},
        .quaternion = {0.0f, 0.0f, 0.0f, 1.0f},
        .scale = {scale.x, scale.y, scale.z},
        .shear = {shear.x, shear.y, shear.z},
        .translation = {translation.x, translation.y, translation.z}};
}

void test_hip_motion_instance_device_ops(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO("Skipping HIP motion-instance device-operation test on '{}'.",
                   device.backend_name());
        return;
    }

    const std::array vertices{
        make_float3(-0.4f, -0.4f, 0.0f),
        make_float3(0.4f, -0.4f, 0.0f),
        make_float3(0.0f, 0.4f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};

    auto stream = device.create_stream();
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);

    AccelMotionOption matrix_option{};
    matrix_option.keyframe_count = 2u;
    matrix_option.mode = AccelMotionMode::MATRIX;
    auto matrix_motion = device.create_motion_instance(mesh, matrix_option);
    std::array matrix_keys{
        translation(make_float3(-3.0f, 0.0f, 0.0f)),
        translation(make_float3(-1.0f, 0.0f, 0.0f))};
    matrix_motion.set_keyframes(luisa::span{matrix_keys});

    AccelMotionOption srt_option{};
    srt_option.keyframe_count = 2u;
    srt_option.mode = AccelMotionMode::SRT;
    auto srt_motion = device.create_motion_instance(mesh, srt_option);
    std::array srt_keys{
        make_srt(make_float3(0.25f, -0.5f, 0.75f),
                 make_float3(1.25f, 0.75f, 1.5f),
                 make_float3(0.1f, -0.2f, 0.3f),
                 make_float3(0.0f, -3.0f, 0.0f)),
        make_srt(make_float3(-0.75f, 0.5f, -0.25f),
                 make_float3(0.8f, 1.4f, 0.6f),
                 make_float3(-0.15f, 0.25f, -0.35f),
                 make_float3(0.0f, -1.0f, 0.0f))};
    srt_motion.set_keyframes(luisa::span{srt_keys});

    auto accel = device.create_accel({.allow_update = true});
    accel.emplace_back(matrix_motion);
    accel.emplace_back(srt_motion);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << matrix_motion.build()
           << srt_motion.build()
           << accel.build()
           << synchronize();

    Kernel1D read_keys = [](AccelVar accel,
                            BufferFloat4x4 matrices,
                            BufferVar<MotionInstanceTransformSRT> srts) noexcept {
        auto key = dispatch_x();
        matrices.write(key, accel.instance_motion_matrix(0u, key));
        srts.write(key, accel.instance_motion_srt(1u, key));
    };
    Kernel1D write_keys = [](AccelVar accel,
                             BufferFloat4x4 matrices,
                             BufferVar<MotionInstanceTransformSRT> srts) noexcept {
        auto key = dispatch_x();
        accel.set_instance_motion_matrix(0u, key, matrices.read(key));
        accel.set_instance_motion_srt(1u, key, srts.read(key));
    };
    Kernel1D trace = [](AccelVar accel, BufferFloat3 origins,
                        BufferUInt results) noexcept {
        auto index = dispatch_x();
        auto ray = make_ray(origins.read(index),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);
        auto hit = accel.intersect_motion(ray, 0.5f, {});
        results.write(index, hit->inst);
    };

    auto read_shader = device.compile(read_keys);
    auto write_shader = device.compile(write_keys);
    auto trace_shader = device.compile(trace);
    auto matrix_buffer = device.create_buffer<float4x4>(2u);
    auto srt_buffer = device.create_buffer<MotionInstanceTransformSRT>(2u);

    auto read_and_check = [&](luisa::span<const float4x4> expected_matrices,
                              luisa::span<const MotionInstanceTransformSRT> expected_srts,
                              luisa::string_view phase) {
        std::array<float4x4, 2u> actual_matrices{};
        std::array<MotionInstanceTransformSRT, 2u> actual_srts{};
        stream << read_shader(accel, matrix_buffer, srt_buffer).dispatch(2u)
               << matrix_buffer.copy_to(luisa::span{actual_matrices})
               << srt_buffer.copy_to(luisa::span{actual_srts})
               << synchronize();
        for (auto i = 0u; i < 2u; i++) {
            expect_matrix(actual_matrices[i], expected_matrices[i],
                          luisa::format("{} matrix key {}", phase, i));
            expect_srt(actual_srts[i], expected_srts[i],
                       luisa::format("{} SRT key {}", phase, i));
        }
    };

    read_and_check(luisa::span{matrix_keys}, luisa::span{srt_keys},
                   "initial device read");

    std::array updated_matrix_keys{
        translation(make_float3(1.0f, 0.0f, 0.0f)),
        translation(make_float3(3.0f, 0.0f, 0.0f))};
    std::array updated_srt_keys{
        make_srt(make_float3(0.1f, 0.2f, 0.3f),
                 make_float3(1.1f, 1.2f, 1.3f),
                 make_float3(0.01f, 0.02f, 0.03f),
                 make_float3(0.0f, 1.0f, 0.0f)),
        make_srt(make_float3(0.1f, 0.2f, 0.3f),
                 make_float3(1.1f, 1.2f, 1.3f),
                 make_float3(0.01f, 0.02f, 0.03f),
                 make_float3(0.0f, 3.0f, 0.0f))};
    stream << matrix_buffer.copy_from(luisa::span{updated_matrix_keys})
           << srt_buffer.copy_from(luisa::span{updated_srt_keys})
           << write_shader(accel, matrix_buffer, srt_buffer).dispatch(2u)
           << accel.build(Accel::BuildRequest::PREFER_UPDATE);

    read_and_check(luisa::span{updated_matrix_keys},
                   luisa::span{updated_srt_keys},
                   "post-refit device read");

    // The nested-scene refit must preserve keyframe timestamps while copying
    // the shader-written affine/SRT payloads into HIPRT's private scene data.
    // At t=0.5, the matrix triangle is centered at x=2 and the SRT triangle at
    // pivot+translation=(0.1, 2.2, 0.3). Their old centers must now miss.
    const std::array origins{
        make_float3(2.0f, 0.0f, 1.0f),
        make_float3(-2.0f, 0.0f, 1.0f),
        make_float3(0.1f, 2.2f, 1.3f),
        make_float3(0.0f, -2.0f, 1.0f)};
    auto origin_buffer = device.create_buffer<float3>(origins.size());
    auto result_buffer = device.create_buffer<uint>(origins.size());
    std::array<uint, origins.size()> results{};
    stream << origin_buffer.copy_from(luisa::span{origins})
           << trace_shader(accel, origin_buffer, result_buffer)
                  .dispatch(origins.size())
           << result_buffer.copy_to(luisa::span{results})
           << synchronize();
    expect(results[0] == 0u) << "matrix setter/refit did not move the triangle";
    expect(results[1] == ~0u) << "matrix setter left stale negative-x bounds";
    expect(results[2] == 1u) << "SRT setter/refit did not move the triangle";
    expect(results[3] == ~0u) << "SRT setter left stale negative-y bounds";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP device motion-instance get/set operations update traversal"_test = [&] {
        test_hip_motion_instance_device_ops(dc->device);
    };
}
