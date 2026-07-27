// Test for exact HIP SRT motion-instance tracing.
// This test covers:
// - T * R * S composition with pivot, all shear terms, nonuniform scale, and translation
// - normalized-linear quaternion interpolation, discriminated from matrix lerp and slerp
// - closest-hit and any-hit traces through an outer transform over a time interval crossing zero
// - conservative rotating bounds at an interior angle missed by the old three-sample bounds
// - motion-instance rebuild propagation after replacing every field of an endpoint keyframe

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct ReferenceSRT {
    float3 pivot{0.0f};
    float4 quaternion{0.0f, 0.0f, 0.0f, 1.0f};
    float3 scale{1.0f};
    float3 shear{0.0f};
    float3 translation{0.0f};
};

struct ExpectedProbe {
    bool hit;
    uint32_t instance;
    uint32_t user_id;
};

constexpr auto primary_instance_index = 1u;
constexpr auto bounds_instance_index = 2u;
constexpr auto primary_user_id = 0x00123456u;
constexpr auto bounds_user_id = 0x00654321u;
constexpr auto expected_primitive_index = 0u;
constexpr auto expected_distance = 1.0f;
constexpr auto expected_barycentric = make_float2(0.25f, 0.5f);
constexpr auto float_tolerance = 1.0e-3f;
constexpr auto motion_time_start = -2.0f;
constexpr auto motion_time_end = 2.0f;

[[nodiscard]] float4 normalize_quaternion(float4 q) noexcept {
    return q / std::sqrt(dot(q, q));
}

[[nodiscard]] float4 quaternion_z(float degrees) noexcept {
    auto half_angle = radians(degrees) * 0.5f;
    return make_float4(0.0f, 0.0f,
                       std::sin(half_angle), std::cos(half_angle));
}

[[nodiscard]] float3 interpolate(float3 lhs, float3 rhs, float alpha) noexcept {
    return lhs + (rhs - lhs) * alpha;
}

[[nodiscard]] float4 quaternion_nlerp(float4 lhs, float4 rhs, float alpha) noexcept {
    lhs = normalize_quaternion(lhs);
    rhs = normalize_quaternion(rhs);
    // The test keys deliberately have positive adjacent dot products, matching
    // the runtime contract and making component-wise normalized lerp unique.
    return normalize_quaternion(lhs + (rhs - lhs) * alpha);
}

[[nodiscard]] float4 quaternion_slerp(float4 lhs, float4 rhs, float alpha) noexcept {
    lhs = normalize_quaternion(lhs);
    rhs = normalize_quaternion(rhs);
    auto cosine = dot(lhs, rhs);
    if (cosine < 0.0f) {
        rhs = -rhs;
        cosine = -cosine;
    }
    cosine = std::clamp(cosine, -1.0f, 1.0f);
    if (cosine > 0.9995f) { return quaternion_nlerp(lhs, rhs, alpha); }
    auto theta = std::acos(cosine);
    auto inverse_sine = 1.0f / std::sin(theta);
    return lhs * (std::sin((1.0f - alpha) * theta) * inverse_sine) +
           rhs * (std::sin(alpha * theta) * inverse_sine);
}

[[nodiscard]] ReferenceSRT interpolate_srt(
    const ReferenceSRT &lhs, const ReferenceSRT &rhs,
    float alpha, bool use_slerp = false) noexcept {
    return ReferenceSRT{
        .pivot = interpolate(lhs.pivot, rhs.pivot, alpha),
        .quaternion = use_slerp ?
                          quaternion_slerp(lhs.quaternion, rhs.quaternion, alpha) :
                          quaternion_nlerp(lhs.quaternion, rhs.quaternion, alpha),
        .scale = interpolate(lhs.scale, rhs.scale, alpha),
        .shear = interpolate(lhs.shear, rhs.shear, alpha),
        .translation = interpolate(lhs.translation, rhs.translation, alpha)};
}

[[nodiscard]] float3 transform_point(const ReferenceSRT &srt, float3 point) noexcept {
    // OptiX/Vulkan SRT semantics use the upper-triangular affine S matrix
    // [sx a b pvx; 0 sy c pvy; 0 0 sz pvz], followed by R and then T.
    auto scaled_sheared = make_float3(
        srt.scale.x * point.x + srt.shear.x * point.y +
            srt.shear.y * point.z + srt.pivot.x,
        srt.scale.y * point.y + srt.shear.z * point.z + srt.pivot.y,
        srt.scale.z * point.z + srt.pivot.z);
    auto q = normalize_quaternion(srt.quaternion);
    auto q_vector = make_float3(q.x, q.y, q.z);
    auto twice_cross = 2.0f * cross(q_vector, scaled_sheared);
    auto rotated = scaled_sheared + q.w * twice_cross +
                   cross(q_vector, twice_cross);
    return rotated + srt.translation;
}

[[nodiscard]] float3 reference_world_point(
    const ReferenceSRT &start, const ReferenceSRT &end,
    float alpha, float3 local_point, float3 outer_translation,
    bool use_slerp = false) noexcept {
    return outer_translation + transform_point(
                                   interpolate_srt(start, end, alpha, use_slerp),
                                   local_point);
}

[[nodiscard]] float3 reference_matrix_lerp_world_point(
    const ReferenceSRT &start, const ReferenceSRT &end,
    float alpha, float3 local_point, float3 outer_translation) noexcept {
    // Applying the linearly interpolated endpoint matrices to one point is
    // equivalent to linearly interpolating the two endpoint point images.
    return outer_translation + interpolate(
                                   transform_point(start, local_point),
                                   transform_point(end, local_point), alpha);
}

[[nodiscard]] MotionInstanceTransformSRT to_runtime_srt(
    const ReferenceSRT &srt) noexcept {
    return MotionInstanceTransformSRT{
        .pivot = {srt.pivot.x, srt.pivot.y, srt.pivot.z},
        .quaternion = {srt.quaternion.x, srt.quaternion.y,
                       srt.quaternion.z, srt.quaternion.w},
        .scale = {srt.scale.x, srt.scale.y, srt.scale.z},
        .shear = {srt.shear.x, srt.shear.y, srt.shear.z},
        .translation = {srt.translation.x, srt.translation.y,
                        srt.translation.z}};
}

[[nodiscard]] float4 make_probe(float3 world_point, float time) noexcept {
    return make_float4(world_point, time);
}

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
            expect(summary.x == expected[i].instance)
                << luisa::format("{} probe {} outer instance: got {}, expected {}",
                                 phase, i, summary.x, expected[i].instance);
            expect(summary.y == expected_primitive_index)
                << luisa::format("{} probe {} primitive: got {}, expected {}",
                                 phase, i, summary.y, expected_primitive_index);
            expect(summary.z == 1u)
                << luisa::format("{} probe {} any-hit trace missed", phase, i);
            expect(summary.w == expected[i].user_id)
                << luisa::format("{} probe {} user ID: got 0x{:08x}, expected 0x{:08x}",
                                 phase, i, summary.w, expected[i].user_id);
            expect_near(phase, i, "distance", detail.x, expected_distance);
            expect_near(phase, i, "barycentric x", detail.y, expected_barycentric.x);
            expect_near(phase, i, "barycentric y", detail.z, expected_barycentric.y);
            expect_near(phase, i, "triangle classification", detail.w, 1.0f);
        } else {
            expect(summary.x == ~0u)
                << luisa::format("{} probe {} closest-hit trace unexpectedly hit instance {}",
                                 phase, i, summary.x);
            expect(summary.z == 0u)
                << luisa::format("{} probe {} any-hit trace unexpectedly hit", phase, i);
            expect(summary.w == ~0u)
                << luisa::format("{} probe {} miss unexpectedly produced user ID 0x{:08x}",
                                 phase, i, summary.w);
            expect_near(phase, i, "miss classification", detail.w, 0.0f);
        }
    }
}

void test_hip_motion_instance_srt(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO("Skipping HIP-specific exact SRT motion-instance test on backend '{}'.",
                   device.backend_name());
        return;
    }

    log_level_verbose();

    const auto primary_local_point = make_float3(4.0f, -2.0f, 1.0f);
    const std::array primary_vertices{
        primary_local_point + make_float3(-0.04f, -0.04f, 0.0f),
        primary_local_point + make_float3(0.04f, -0.04f, 0.0f),
        primary_local_point + make_float3(0.0f, 0.04f, 0.0f)};
    const auto bounds_local_point = make_float3(20.0f, 0.0f, 0.0f);
    const std::array bounds_vertices{
        bounds_local_point + make_float3(-0.04f, -0.04f, 0.0f),
        bounds_local_point + make_float3(0.04f, -0.04f, 0.0f),
        bounds_local_point + make_float3(0.0f, 0.04f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};

    const ReferenceSRT primary_start{
        .pivot = make_float3(0.25f, -0.4f, 0.3f),
        .quaternion = quaternion_z(0.0f),
        .scale = make_float3(1.2f, 0.8f, 1.5f),
        .shear = make_float3(0.4f, -0.4f, 0.5f),
        .translation = make_float3(-0.5f, 0.75f, -0.2f)};
    const ReferenceSRT primary_end{
        .pivot = make_float3(-0.35f, 0.2f, -0.1f),
        .quaternion = quaternion_z(170.0f),
        .scale = make_float3(0.7f, 1.6f, 0.9f),
        .shear = make_float3(-0.2f, 0.4f, -0.3f),
        .translation = make_float3(1.0f, -0.6f, 0.5f)};
    const ReferenceSRT primary_rebuilt_end{
        .pivot = make_float3(0.6f, -0.1f, 0.45f),
        .quaternion = quaternion_z(-130.0f),
        .scale = make_float3(1.5f, 0.6f, 1.2f),
        .shear = make_float3(-0.35f, 0.15f, 0.4f),
        .translation = make_float3(-1.2f, 0.3f, -0.6f)};
    const auto primary_outer_translation = make_float3(6.0f, -4.0f, 1.25f);

    const ReferenceSRT bounds_start{};
    auto bounds_end = bounds_start;
    bounds_end.quaternion = quaternion_z(150.0f);
    const auto bounds_outer_translation = make_float3(-30.0f, 5.0f, -1.0f);

    auto stream = device.create_stream();
    auto primary_vertex_buffer = device.create_buffer<float3>(primary_vertices.size());
    auto bounds_vertex_buffer = device.create_buffer<float3>(bounds_vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto primary_mesh = device.create_mesh(primary_vertex_buffer, triangle_buffer);
    auto bounds_mesh = device.create_mesh(bounds_vertex_buffer, triangle_buffer);

    AccelMotionOption motion_option{};
    motion_option.keyframe_count = 2u;
    motion_option.time_start = motion_time_start;
    motion_option.time_end = motion_time_end;
    motion_option.mode = AccelMotionMode::SRT;
    // Vanish behavior is intentionally outside this focused interpolation and
    // conservative-bounds test.
    auto primary_motion = device.create_motion_instance(primary_mesh, motion_option);
    auto bounds_motion = device.create_motion_instance(bounds_mesh, motion_option);
    std::array primary_keyframes{
        to_runtime_srt(primary_start),
        to_runtime_srt(primary_end)};
    const std::array bounds_keyframes{
        to_runtime_srt(bounds_start),
        to_runtime_srt(bounds_end)};
    primary_motion.set_keyframes(luisa::span{primary_keyframes});
    bounds_motion.set_keyframes(luisa::span{bounds_keyframes});

    auto accel = device.create_accel({.allow_update = true});
    accel.emplace_back(primary_mesh, translation(-100.0f, 0.0f, 0.0f),
                       0xffu, true, 7u);
    accel.emplace_back(primary_motion, translation(primary_outer_translation),
                       0xffu, true, primary_user_id);
    accel.emplace_back(bounds_motion, translation(bounds_outer_translation),
                       0xffu, true, bounds_user_id);

    stream << primary_vertex_buffer.copy_from(luisa::span{primary_vertices})
           << bounds_vertex_buffer.copy_from(luisa::span{bounds_vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << primary_mesh.build()
           << bounds_mesh.build()
           << primary_motion.build()
           << bounds_motion.build()
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
    trace_motion.function_builder()->set_name("hip_srt_motion_trace_callable");

    Kernel1D trace = [&trace_motion](AccelVar accel, BufferFloat4 probes,
                                     BufferUInt4 summaries,
                                     BufferFloat4 details) noexcept {
        auto index = dispatch_id().x;
        auto probe = probes.read(index);
        auto ray = make_ray(make_float3(probe.x, probe.y, probe.z + 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);
        trace_motion(accel, ray, probe.w, index, summaries, details);
    };

    auto shader = device.compile(trace);
    constexpr auto probe_capacity = 7u;
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

    constexpr auto quarter_alpha = 0.25f;
    constexpr auto midpoint_alpha = 0.5f;
    constexpr auto quarter_time = -1.0f;
    constexpr auto midpoint_time = 0.0f;
    const auto initial_start = reference_world_point(
        primary_start, primary_end, 0.0f, primary_local_point,
        primary_outer_translation);
    const auto initial_quarter = reference_world_point(
        primary_start, primary_end, quarter_alpha, primary_local_point,
        primary_outer_translation);
    const auto initial_midpoint = reference_world_point(
        primary_start, primary_end, midpoint_alpha, primary_local_point,
        primary_outer_translation);
    const auto initial_end = reference_world_point(
        primary_start, primary_end, 1.0f, primary_local_point,
        primary_outer_translation);
    const auto slerp_only_quarter = reference_world_point(
        primary_start, primary_end, quarter_alpha, primary_local_point,
        primary_outer_translation, true);
    const auto matrix_lerp_only_quarter = reference_matrix_lerp_world_point(
        primary_start, primary_end, quarter_alpha, primary_local_point,
        primary_outer_translation);

    // q0=identity and q1=Rz(150 degrees). Nlerp reaches exactly 90 degrees at
    // alpha=2-sqrt(2), between the old alpha=0.5 and 0.75 bound samples. The
    // radius-20 triangle reaches y=20 there, beyond their sampled maximum.
    constexpr auto bounds_alpha = 0.585786437626905f;
    constexpr auto bounds_time = motion_time_start +
                                 (motion_time_end - motion_time_start) * bounds_alpha;
    const auto bounds_interior = reference_world_point(
        bounds_start, bounds_end, bounds_alpha, bounds_local_point,
        bounds_outer_translation);

    const std::array initial_probes{
        make_probe(initial_start, motion_time_start),
        make_probe(initial_quarter, quarter_time),
        make_probe(initial_midpoint, midpoint_time),
        make_probe(initial_end, motion_time_end),
        make_probe(slerp_only_quarter, quarter_time),
        make_probe(matrix_lerp_only_quarter, quarter_time),
        make_probe(bounds_interior, bounds_time)};
    const std::array initial_expected{
        ExpectedProbe{true, primary_instance_index, primary_user_id},
        ExpectedProbe{true, primary_instance_index, primary_user_id},
        ExpectedProbe{true, primary_instance_index, primary_user_id},
        ExpectedProbe{true, primary_instance_index, primary_user_id},
        ExpectedProbe{false, ~0u, ~0u},
        ExpectedProbe{false, ~0u, ~0u},
        ExpectedProbe{true, bounds_instance_index, bounds_user_id}};
    run(luisa::span{initial_probes}, luisa::span{initial_expected},
        "initial exact SRT build");

    primary_keyframes[1] = to_runtime_srt(primary_rebuilt_end);
    primary_motion.set_keyframes(luisa::span{primary_keyframes});
    stream << primary_motion.build()
           << accel.build(AccelBuildRequest::FORCE_BUILD)
           << synchronize();

    const auto rebuilt_quarter = reference_world_point(
        primary_start, primary_rebuilt_end, quarter_alpha,
        primary_local_point, primary_outer_translation);
    const auto rebuilt_midpoint = reference_world_point(
        primary_start, primary_rebuilt_end, midpoint_alpha,
        primary_local_point, primary_outer_translation);
    const auto rebuilt_end = reference_world_point(
        primary_start, primary_rebuilt_end, 1.0f,
        primary_local_point, primary_outer_translation);
    const std::array rebuilt_probes{
        make_probe(initial_start, motion_time_start),
        make_probe(rebuilt_quarter, quarter_time),
        make_probe(rebuilt_midpoint, midpoint_time),
        make_probe(rebuilt_end, motion_time_end),
        make_probe(initial_quarter, quarter_time),
        make_probe(initial_end, motion_time_end),
        make_probe(bounds_interior, bounds_time)};
    const std::array rebuilt_expected{
        ExpectedProbe{true, primary_instance_index, primary_user_id},
        ExpectedProbe{true, primary_instance_index, primary_user_id},
        ExpectedProbe{true, primary_instance_index, primary_user_id},
        ExpectedProbe{true, primary_instance_index, primary_user_id},
        ExpectedProbe{false, ~0u, ~0u},
        ExpectedProbe{false, ~0u, ~0u},
        ExpectedProbe{true, bounds_instance_index, bounds_user_id}};
    run(luisa::span{rebuilt_probes}, luisa::span{rebuilt_expected},
        "rebuilt exact SRT endpoint");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP SRT motion instances preserve exact interpolation and bounds"_test = [&] {
        test_hip_motion_instance_srt(dc->device);
    };
}
