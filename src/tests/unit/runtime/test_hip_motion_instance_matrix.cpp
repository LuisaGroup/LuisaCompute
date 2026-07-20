// Test for exact HIP MATRIX motion-instance tracing.
// This test covers:
// - direct closest-hit and any-hit motion traces at both endpoints and the midpoint
// - transitive motion-use analysis and traversal-stack ABI propagation through a callable
// - element-wise MATRIX interpolation rather than SRT/quaternion interpolation
// - composition with an outer TLAS transform and preservation of the outer instance/user ID
// - motion-instance rebuild propagation after replacing an endpoint keyframe

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>
#include <cmath>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct ExpectedProbe {
    bool hit;
};

constexpr auto motion_instance_index = 1u;
constexpr auto motion_instance_user_id = 0x0055aa11u;
constexpr auto expected_primitive_index = 0u;
constexpr auto expected_distance = 1.0f;
constexpr auto expected_barycentric = make_float2(0.25f, 0.5f);
constexpr auto float_tolerance = 2.0e-4f;

void expect_near(luisa::string_view phase, size_t probe_index,
                 luisa::string_view field, float actual, float expected) {
    expect(std::abs(actual - expected) < float_tolerance)
        << luisa::format("{} probe {} {}: got {}, expected {}",
                         phase, probe_index, field, actual, expected);
}

void check_results(luisa::string_view phase,
                   luisa::span<const uint4> summaries,
                   luisa::span<const float4> details,
                   luisa::span<const ExpectedProbe> expected) {
    expect(summaries.size() == expected.size());
    expect(details.size() == expected.size());
    for (auto i = 0u; i < expected.size(); i++) {
        auto summary = summaries[i];
        auto detail = details[i];
        if (expected[i].hit) {
            expect(summary.x == motion_instance_index)
                << luisa::format("{} probe {} outer instance: got {}, expected {}",
                                 phase, i, summary.x, motion_instance_index);
            expect(summary.y == expected_primitive_index)
                << luisa::format("{} probe {} primitive: got {}, expected {}",
                                 phase, i, summary.y, expected_primitive_index);
            expect(summary.z == 1u)
                << luisa::format("{} probe {} any-hit trace missed",
                                 phase, i);
            expect(summary.w == motion_instance_user_id)
                << luisa::format("{} probe {} user ID: got 0x{:08x}, expected 0x{:08x}",
                                 phase, i, summary.w, motion_instance_user_id);
            expect_near(phase, i, "distance", detail.x, expected_distance);
            expect_near(phase, i, "barycentric x", detail.y, expected_barycentric.x);
            expect_near(phase, i, "barycentric y", detail.z, expected_barycentric.y);
            expect_near(phase, i, "triangle classification", detail.w, 1.0f);
        } else {
            expect(summary.x == ~0u)
                << luisa::format("{} probe {} closest-hit trace unexpectedly hit instance {}",
                                 phase, i, summary.x);
            expect(summary.z == 0u)
                << luisa::format("{} probe {} any-hit trace unexpectedly hit",
                                 phase, i);
            expect(summary.w == ~0u)
                << luisa::format("{} probe {} miss unexpectedly produced user ID 0x{:08x}",
                                 phase, i, summary.w);
            expect_near(phase, i, "miss classification", detail.w, 0.0f);
        }
    }
}

void test_hip_motion_instance_matrix(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO("Skipping HIP-specific MATRIX motion-instance test on backend '{}'.",
                   device.backend_name());
        return;
    }

    log_level_verbose();

    // The local point (2, 0, 0) is strictly inside this triangle, with
    // barycentric coordinates (0.25, 0.5). At t=0.5, linear interpolation
    // between I and Rz(120 degrees) maps it to 0.5 * Rz(60 degrees) * p.
    const std::array vertices{
        make_float3(1.5f, -0.5f, 0.0f),
        make_float3(2.5f, -0.5f, 0.0f),
        make_float3(2.0f, 0.5f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};

    auto stream = device.create_stream();
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);

    AccelMotionOption motion_option{};
    motion_option.keyframe_count = 2u;
    motion_option.time_start = 0.0f;
    motion_option.time_end = 1.0f;
    motion_option.mode = AccelMotionMode::MATRIX;
    auto motion_instance = device.create_motion_instance(mesh, motion_option);

    std::array keyframes{
        make_float4x4(1.0f),
        rotation(make_float3(0.0f, 0.0f, 1.0f), radians(120.0f))};
    motion_instance.set_keyframes(luisa::span{keyframes});

    auto accel = device.create_accel({.allow_update = true});
    // The decoy makes the expected outer instance index observably different
    // from the nested motion scene's sole child index.
    accel.emplace_back(mesh, translation(-100.0f, 0.0f, 0.0f),
                       0xffu, true, 11u);
    // Translation does not commute with the inner rotation, so the probe
    // positions also verify outer-after-inner transform composition.
    accel.emplace_back(motion_instance, translation(3.0f, -2.0f, 0.0f),
                       0xffu, true, motion_instance_user_id);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << motion_instance.build()
           << accel.build()
           << synchronize();

    Callable trace_motion = [](AccelVar accel, Var<Ray> ray, Float time,
                               UInt index, BufferUInt4 summaries,
                               BufferFloat4 details) noexcept {
        auto closest = accel.intersect_motion(ray, time, {});
        auto any = accel.intersect_any_motion(ray, time, {});
        UInt user_id = ~0u;
        $if (!closest->miss()) {
            user_id = accel.instance_user_id(closest->inst);
        };

        summaries.write(index, make_uint4(
                                   closest->inst,
                                   closest->prim,
                                   cast<uint>(any),
                                   user_id));
        details.write(index, make_float4(
                                 closest->distance(),
                                 closest->bary.x,
                                 closest->bary.y,
                                 cast<float>(closest->is_triangle())));
    };
    trace_motion.function_builder()->set_name("hip_motion_trace_callable");

    Kernel1D trace = [&trace_motion](AccelVar accel, BufferFloat4 probes,
                                     BufferUInt4 summaries,
                                     BufferFloat4 details) noexcept {
        auto index = dispatch_id().x;
        auto probe = probes.read(index);
        auto ray = make_ray(make_float3(probe.x, probe.y, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);
        trace_motion(accel, ray, probe.z, index, summaries, details);
    };

    auto shader = device.compile(trace);
    constexpr auto probe_capacity = 5u;
    auto probe_buffer = device.create_buffer<float4>(probe_capacity);
    auto summary_buffer = device.create_buffer<uint4>(probe_capacity);
    auto detail_buffer = device.create_buffer<float4>(probe_capacity);

    auto run = [&](luisa::span<const float4> probes,
                   luisa::span<const ExpectedProbe> expected,
                   luisa::string_view phase) {
        expect(probes.size() == expected.size());
        expect(probes.size() <= probe_capacity);
        luisa::vector<uint4> host_summaries(probes.size());
        luisa::vector<float4> host_details(probes.size());
        stream << probe_buffer.view(0u, probes.size()).copy_from(probes)
               << shader(accel, probe_buffer, summary_buffer, detail_buffer).dispatch(probes.size())
               << summary_buffer.view(0u, probes.size()).copy_to(luisa::span{host_summaries})
               << detail_buffer.view(0u, probes.size()).copy_to(luisa::span{host_details})
               << synchronize();
        check_results(phase, luisa::span{host_summaries},
                      luisa::span{host_details}, expected);
    };

    // Correct element-wise MATRIX interpolation maps the local point to
    // (3.5, -1.1339746) after the outer transform. Quaternion/SRT
    // interpolation would map it to (4, -0.2679492), which must miss.
    const std::array initial_probes{
        make_float4(5.0f, -2.0f, 0.0f, 0.0f),
        make_float4(3.5f, -1.133974596f, 0.5f, 0.0f),
        make_float4(4.0f, -0.267949192f, 0.5f, 0.0f),
        make_float4(2.0f, -0.267949192f, 1.0f, 0.0f),
        make_float4(3.5f, -2.866025404f, 0.5f, 0.0f)};
    const std::array initial_expected{
        ExpectedProbe{true},
        ExpectedProbe{true},
        ExpectedProbe{false},
        ExpectedProbe{true},
        ExpectedProbe{false}};
    run(luisa::span{initial_probes}, luisa::span{initial_expected},
        "positive endpoint build");

    // Replace Rz(+120 degrees) with Rz(-120 degrees). Rebuilding both the
    // nested motion scene and outer TLAS must move the midpoint and endpoint
    // to the lower half-plane and retire the old positive midpoint.
    keyframes[1] = rotation(make_float3(0.0f, 0.0f, 1.0f), radians(-120.0f));
    motion_instance.set_keyframes(luisa::span{keyframes});
    stream << motion_instance.build()
           << accel.build(AccelBuildRequest::FORCE_BUILD)
           << synchronize();

    const std::array rebuilt_probes{
        make_float4(5.0f, -2.0f, 0.0f, 0.0f),
        make_float4(3.5f, -2.866025404f, 0.5f, 0.0f),
        make_float4(4.0f, -3.732050808f, 0.5f, 0.0f),
        make_float4(2.0f, -3.732050808f, 1.0f, 0.0f),
        make_float4(3.5f, -1.133974596f, 0.5f, 0.0f)};
    const std::array rebuilt_expected{
        ExpectedProbe{true},
        ExpectedProbe{true},
        ExpectedProbe{false},
        ExpectedProbe{true},
        ExpectedProbe{false}};
    run(luisa::span{rebuilt_probes}, luisa::span{rebuilt_expected},
        "negative endpoint rebuild");
}

void test_hip_motion_instance_curve(Device &device) {
    constexpr auto curve_basis = CurveBasis::PIECEWISE_LINEAR;
    const std::array control_points{
        make_float4(-0.5f, 0.0f, 0.0f, 0.1f),
        make_float4(0.5f, 0.0f, 0.0f, 0.1f)};
    const std::array segments{0u};

    auto stream = device.create_stream();
    auto control_point_buffer = device.create_buffer<float4>(control_points.size());
    auto segment_buffer = device.create_buffer<uint>(segments.size());
    auto curve = device.create_curve(
        curve_basis, control_point_buffer, segment_buffer);

    AccelMotionOption motion_option{};
    motion_option.keyframe_count = 2u;
    motion_option.time_start = 0.0f;
    motion_option.time_end = 1.0f;
    motion_option.mode = AccelMotionMode::MATRIX;
    auto motion_instance = device.create_motion_instance(curve, motion_option);
    const std::array keyframes{
        make_float4x4(1.0f),
        translation(0.0f, 1.0f, 0.0f)};
    motion_instance.set_keyframes(luisa::span{keyframes});

    auto accel = device.create_accel();
    accel.emplace_back(motion_instance, translation(3.0f, 0.0f, 0.0f),
                       0xffu, true, motion_instance_user_id);
    stream << control_point_buffer.copy_from(luisa::span{control_points})
           << segment_buffer.copy_from(luisa::span{segments})
           << curve.build()
           << motion_instance.build()
           << accel.build()
           << synchronize();

    Kernel1D trace = [](AccelVar accel, BufferUInt4 summary,
                        BufferFloat4 detail) noexcept {
        auto ray = make_ray(make_float3(3.0f, 0.5f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);
        auto options = AccelTraceOptions{
            .curve_bases = {CurveBasis::PIECEWISE_LINEAR}};
        auto closest = accel.intersect_motion(ray, 0.5f, options);
        auto any = accel.intersect_any_motion(ray, 0.5f, options);
        UInt user_id = ~0u;
        $if (!closest->miss()) {
            user_id = accel.instance_user_id(closest->inst);
        };
        summary.write(0u, make_uint4(
                              closest->inst, closest->prim,
                              cast<uint>(any), user_id));
        detail.write(0u, make_float4(
                             closest->distance(), closest->curve_parameter(),
                             closest->bary.y,
                             cast<float>(closest->is_curve())));
    };

    auto shader = device.compile(trace);
    auto summary_buffer = device.create_buffer<uint4>(1u);
    auto detail_buffer = device.create_buffer<float4>(1u);
    uint4 summary{};
    float4 detail{};
    stream << shader(accel, summary_buffer, detail_buffer).dispatch(1u)
           << summary_buffer.copy_to(luisa::span{&summary, 1u})
           << detail_buffer.copy_to(luisa::span{&detail, 1u})
           << synchronize();

    expect(summary.x == 0u) << "motion curve returned the wrong outer instance";
    expect(summary.y == 0u) << "motion curve returned the wrong primitive";
    expect(summary.z == 1u) << "motion curve any-hit trace missed";
    expect(summary.w == motion_instance_user_id)
        << "motion curve lost outer instance metadata";
    expect_near("motion curve", 0u, "distance", detail.x, 0.9f);
    expect_near("motion curve", 0u, "curve parameter", detail.y, 0.5f);
    expect_near("motion curve", 0u, "curve marker", detail.z, -1.0f);
    expect_near("motion curve", 0u, "curve classification", detail.w, 1.0f);
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP MATRIX motion instances preserve exact nested trace semantics"_test = [&] {
        test_hip_motion_instance_matrix(dc->device);
    };
    "HIP MATRIX motion instances preserve direct curve intersections"_test = [&] {
        if (dc->device.backend_name() == "hip") {
            test_hip_motion_instance_curve(dc->device);
        }
    };
}
