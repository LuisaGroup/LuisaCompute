// Exact HIP coverage for static traversal through motion instances.
// This test covers:
// - implicit time zero for MATRIX and SRT motion-instance interpolation
// - static closest/any traces and ALL/ANY ray queries
// - both gfx12's static-only flat path and motion-query-forced generic path
// - preservation of outer TLAS instance/user IDs, primitive IDs, distance, and barycentrics
// - a deforming motion-mesh child nested below a motion instance
// - exact rejection of swept bounds at the child-only and instance-only start poses

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

struct StaticMotionProbeResult {
    uint4 direct;
    float4 direct_detail;
    uint4 all_committed;
    uint4 all_callback;
    float4 all_committed_detail;
    float4 all_callback_detail;
    uint4 any_committed;
    uint4 any_callback;
    float4 any_committed_detail;
    float4 any_callback_detail;
};
LUISA_STRUCT(StaticMotionProbeResult,
             direct, direct_detail,
             all_committed, all_callback,
             all_committed_detail, all_callback_detail,
             any_committed, any_callback,
             any_committed_detail, any_callback_detail) {};

namespace {

struct ExpectedProbe {
    luisa::string_view name;
    bool hit;
    uint32_t instance;
    uint32_t user_id;
};

constexpr auto kPrimitiveIndex = 0u;
constexpr auto kExpectedDistance = 1.0f;
constexpr auto kExpectedBarycentric = make_float2(0.25f, 0.5f);
constexpr auto kTolerance = 2.0e-4f;

[[nodiscard]] MotionInstanceTransformSRT make_translation_srt(float3 translation) noexcept {
    return MotionInstanceTransformSRT{
        .pivot = {0.0f, 0.0f, 0.0f},
        .quaternion = {0.0f, 0.0f, 0.0f, 1.0f},
        .scale = {1.0f, 1.0f, 1.0f},
        .shear = {0.0f, 0.0f, 0.0f},
        .translation = {translation.x, translation.y, translation.z}};
}

void expect_near(luisa::string_view path, luisa::string_view probe,
                 luisa::string_view field, float actual, float expected) {
    expect(std::abs(actual - expected) < kTolerance)
        << luisa::format("{} probe '{}' {}: got {}, expected {}",
                         path, probe, field, actual, expected);
}

void check_hit_detail(luisa::string_view path, luisa::string_view probe,
                      float4 detail) {
    expect_near(path, probe, "distance", detail.x, kExpectedDistance);
    expect_near(path, probe, "barycentric x", detail.y,
                kExpectedBarycentric.x);
    expect_near(path, probe, "barycentric y", detail.z,
                kExpectedBarycentric.y);
    expect_near(path, probe, "triangle classification", detail.w, 1.0f);
}

void check_committed(luisa::string_view path, const ExpectedProbe &expected,
                     uint4 summary, float4 detail) {
    if (expected.hit) {
        expect(summary.x == expected.instance)
            << luisa::format("{} probe '{}' instance: got {}, expected {}",
                             path, expected.name, summary.x, expected.instance);
        expect(summary.y == kPrimitiveIndex)
            << luisa::format("{} probe '{}' primitive: got {}, expected {}",
                             path, expected.name, summary.y, kPrimitiveIndex);
        expect(summary.z == expected.user_id)
            << luisa::format("{} probe '{}' user ID: got {}, expected {}",
                             path, expected.name, summary.z, expected.user_id);
        expect(summary.w == 1u)
            << luisa::format("{} probe '{}' was not classified as a triangle",
                             path, expected.name);
        check_hit_detail(path, expected.name, detail);
    } else {
        expect(summary.x == ~0u)
            << luisa::format("{} probe '{}' unexpectedly hit instance {}",
                             path, expected.name, summary.x);
        expect(summary.y == ~0u)
            << luisa::format("{} probe '{}' miss returned primitive {}",
                             path, expected.name, summary.y);
        expect(summary.z == ~0u)
            << luisa::format("{} probe '{}' miss returned user ID {}",
                             path, expected.name, summary.z);
        expect(summary.w == 0u)
            << luisa::format("{} probe '{}' miss was classified as a triangle",
                             path, expected.name);
        expect_near(path, expected.name, "miss classification",
                    detail.w, 0.0f);
    }
}

void check_callback(luisa::string_view path, const ExpectedProbe &expected,
                    uint4 summary, float4 detail) {
    if (expected.hit) {
        expect(summary.x == 1u)
            << luisa::format("{} probe '{}' callback count: got {}, expected 1",
                             path, expected.name, summary.x);
        expect(summary.y == expected.instance)
            << luisa::format("{} probe '{}' callback instance: got {}, expected {}",
                             path, expected.name, summary.y, expected.instance);
        expect(summary.z == kPrimitiveIndex)
            << luisa::format("{} probe '{}' callback primitive: got {}, expected {}",
                             path, expected.name, summary.z, kPrimitiveIndex);
        expect(summary.w == expected.user_id)
            << luisa::format("{} probe '{}' callback user ID: got {}, expected {}",
                             path, expected.name, summary.w, expected.user_id);
        check_hit_detail(path, expected.name, detail);
    } else {
        expect(summary.x == 0u)
            << luisa::format("{} probe '{}' reached {} callbacks; exact traversal should reject it",
                             path, expected.name, summary.x);
        expect(summary.y == ~0u)
            << luisa::format("{} probe '{}' unexpectedly reported callback instance {}",
                             path, expected.name, summary.y);
        expect(summary.z == ~0u)
            << luisa::format("{} probe '{}' unexpectedly reported callback primitive {}",
                             path, expected.name, summary.z);
        expect(summary.w == ~0u)
            << luisa::format("{} probe '{}' unexpectedly reported callback user ID {}",
                             path, expected.name, summary.w);
        expect_near(path, expected.name, "untouched callback distance",
                    detail.x, -1.0f);
        expect_near(path, expected.name, "callback miss classification",
                    detail.w, 0.0f);
    }
}

void check_result(const ExpectedProbe &expected,
                  const StaticMotionProbeResult &result) {
    if (expected.hit) {
        expect(result.direct.x == expected.instance)
            << luisa::format("direct probe '{}' instance: got {}, expected {}",
                             expected.name, result.direct.x, expected.instance);
        expect(result.direct.y == kPrimitiveIndex)
            << luisa::format("direct probe '{}' primitive: got {}, expected {}",
                             expected.name, result.direct.y, kPrimitiveIndex);
        expect(result.direct.z == 1u)
            << luisa::format("direct probe '{}' any-hit trace missed",
                             expected.name);
        expect(result.direct.w == expected.user_id)
            << luisa::format("direct probe '{}' user ID: got {}, expected {}",
                             expected.name, result.direct.w, expected.user_id);
        check_hit_detail("direct", expected.name, result.direct_detail);
    } else {
        expect(result.direct.x == ~0u)
            << luisa::format("direct probe '{}' unexpectedly hit instance {}",
                             expected.name, result.direct.x);
        expect(result.direct.y == ~0u)
            << luisa::format("direct probe '{}' miss returned primitive {}",
                             expected.name, result.direct.y);
        expect(result.direct.z == 0u)
            << luisa::format("direct probe '{}' any-hit trace unexpectedly hit",
                             expected.name);
        expect(result.direct.w == ~0u)
            << luisa::format("direct probe '{}' miss returned user ID {}",
                             expected.name, result.direct.w);
        expect_near("direct", expected.name, "miss classification",
                    result.direct_detail.w, 0.0f);
    }

    check_committed("static ALL query committed", expected,
                    result.all_committed, result.all_committed_detail);
    check_callback("static ALL query callback", expected,
                   result.all_callback, result.all_callback_detail);
    check_committed("static ANY query committed", expected,
                    result.any_committed, result.any_committed_detail);
    check_callback("static ANY query callback", expected,
                   result.any_callback, result.any_callback_detail);
}

void test_hip_static_motion_instance(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO("Skipping HIP-specific static motion-instance test on backend '{}'.",
                   device.backend_name());
        return;
    }

    log_level_verbose();

    // At barycentrics (u, v) = (0.25, 0.5), the target is exactly (0, 0, 0).
    // The nonzero z extent also keeps custom swept AABBs nondegenerate.
    const std::array static_vertices{
        make_float3(-0.5f, -0.5f, -0.05f),
        make_float3(0.5f, -0.5f, -0.05f),
        make_float3(0.0f, 0.5f, 0.05f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    const std::array deforming_vertices{
        // keyframe 0: target x = -3
        make_float3(-3.5f, -0.5f, -0.05f),
        make_float3(-2.5f, -0.5f, -0.05f),
        make_float3(-3.0f, 0.5f, 0.05f),
        // keyframe 1: target x = +3
        make_float3(2.5f, -0.5f, -0.05f),
        make_float3(3.5f, -0.5f, -0.05f),
        make_float3(3.0f, 0.5f, 0.05f)};

    auto stream = device.create_stream();
    auto static_vertex_buffer =
        device.create_buffer<float3>(static_vertices.size());
    auto deforming_vertex_buffer =
        device.create_buffer<float3>(deforming_vertices.size());
    auto static_triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    auto deforming_triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    auto static_mesh = device.create_mesh(
        static_vertex_buffer, static_triangle_buffer);

    AccelOption deforming_mesh_option{};
    deforming_mesh_option.motion.keyframe_count = 2u;
    deforming_mesh_option.motion.time_start = -1.0f;
    deforming_mesh_option.motion.time_end = 1.0f;
    auto deforming_mesh = device.create_mesh(
        deforming_vertex_buffer, deforming_triangle_buffer,
        deforming_mesh_option);

    AccelMotionOption matrix_option{};
    matrix_option.keyframe_count = 2u;
    matrix_option.time_start = -2.0f;
    matrix_option.time_end = 2.0f;
    matrix_option.mode = AccelMotionMode::MATRIX;
    auto matrix_motion =
        device.create_motion_instance(static_mesh, matrix_option);
    const std::array matrix_keyframes{
        translation(-2.0f, 0.0f, 0.0f),
        translation(2.0f, 0.0f, 0.0f)};
    matrix_motion.set_keyframes(luisa::span{matrix_keyframes});

    AccelMotionOption srt_option{};
    srt_option.keyframe_count = 2u;
    srt_option.time_start = -4.0f;
    srt_option.time_end = 4.0f;
    srt_option.mode = AccelMotionMode::SRT;
    auto srt_motion = device.create_motion_instance(static_mesh, srt_option);
    const std::array srt_keyframes{
        make_translation_srt(make_float3(0.0f, -2.0f, 0.0f)),
        make_translation_srt(make_float3(0.0f, 2.0f, 0.0f))};
    srt_motion.set_keyframes(luisa::span{srt_keyframes});

    AccelMotionOption nested_option{};
    nested_option.keyframe_count = 2u;
    nested_option.time_start = -2.0f;
    nested_option.time_end = 2.0f;
    nested_option.mode = AccelMotionMode::MATRIX;
    auto nested_motion =
        device.create_motion_instance(deforming_mesh, nested_option);
    const std::array nested_keyframes{
        translation(0.0f, -2.0f, 0.0f),
        translation(0.0f, 2.0f, 0.0f)};
    nested_motion.set_keyframes(luisa::span{nested_keyframes});

    constexpr auto matrix_instance = 1u;
    constexpr auto srt_instance = 2u;
    constexpr auto nested_instance = 3u;
    constexpr auto matrix_user_id = 101u;
    constexpr auto srt_user_id = 202u;
    constexpr auto nested_user_id = 303u;

    auto accel = device.create_accel();
    // Reserve instance zero with a decoy so nested-scene traversal must
    // preserve the observable outer TLAS instance index.
    accel.emplace_back(static_mesh, translation(-100.0f, 0.0f, 0.0f),
                       0xffu, true, 1u);
    accel.emplace_back(matrix_motion, translation(5.0f, 0.0f, 0.0f),
                       0xffu, false, matrix_user_id);
    accel.emplace_back(srt_motion, translation(10.0f, 3.0f, 0.0f),
                       0xffu, false, srt_user_id);
    accel.emplace_back(nested_motion, translation(15.0f, 6.0f, 0.0f),
                       0xffu, false, nested_user_id);

    stream << static_vertex_buffer.copy_from(luisa::span{static_vertices})
           << deforming_vertex_buffer.copy_from(luisa::span{deforming_vertices})
           << static_triangle_buffer.copy_from(luisa::span{triangles})
           << deforming_triangle_buffer.copy_from(luisa::span{triangles})
           << static_mesh.build()
           << deforming_mesh.build()
           << matrix_motion.build()
           << srt_motion.build()
           << nested_motion.build()
           << accel.build()
           << synchronize();

    auto evaluate_static_paths = [](const AccelVar &accel,
                                    const Var<Ray> &ray) noexcept {
        Var<StaticMotionProbeResult> result;
        auto closest = accel.intersect(ray, {});
        auto any = accel.intersect_any(ray, {});
        UInt closest_user_id = ~0u;
        $if (!closest->miss()) {
            closest_user_id = accel.instance_user_id(closest->inst);
        };
        result.direct = make_uint4(
            closest->inst, closest->prim,
            cast<uint>(any), closest_user_id);
        result.direct_detail = make_float4(
            closest->distance(), closest->bary.x, closest->bary.y,
            cast<float>(closest->is_triangle()));

        UInt all_callback_count = 0u;
        UInt all_callback_instance = ~0u;
        UInt all_callback_primitive = ~0u;
        UInt all_callback_user_id = ~0u;
        Float4 all_callback_detail = make_float4(-1.0f, -1.0f, -1.0f, 0.0f);
        auto all_committed = accel.traverse(ray, {})
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         all_callback_count += 1u;
                                         all_callback_instance = hit->inst;
                                         all_callback_primitive = hit->prim;
                                         all_callback_user_id = accel.instance_user_id(hit->inst);
                                         all_callback_detail = make_float4(
                                             hit->distance(), hit->bary.x, hit->bary.y,
                                             cast<float>(hit->is_triangle()));
                                         candidate.commit();
                                     })
                                 .trace();
        UInt all_committed_user_id = ~0u;
        $if (!all_committed->miss()) {
            all_committed_user_id = accel.instance_user_id(all_committed->inst);
        };
        result.all_committed = make_uint4(
            all_committed->inst, all_committed->prim,
            all_committed_user_id,
            cast<uint>(all_committed->is_triangle()));
        result.all_callback = make_uint4(
            all_callback_count, all_callback_instance,
            all_callback_primitive, all_callback_user_id);
        result.all_committed_detail = make_float4(
            all_committed->distance(), all_committed->bary.x,
            all_committed->bary.y,
            cast<float>(all_committed->is_triangle()));
        result.all_callback_detail = all_callback_detail;

        UInt any_callback_count = 0u;
        UInt any_callback_instance = ~0u;
        UInt any_callback_primitive = ~0u;
        UInt any_callback_user_id = ~0u;
        Float4 any_callback_detail = make_float4(-1.0f, -1.0f, -1.0f, 0.0f);
        auto any_committed = accel.traverse_any(ray, {})
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         any_callback_count += 1u;
                                         any_callback_instance = hit->inst;
                                         any_callback_primitive = hit->prim;
                                         any_callback_user_id = accel.instance_user_id(hit->inst);
                                         any_callback_detail = make_float4(
                                             hit->distance(), hit->bary.x, hit->bary.y,
                                             cast<float>(hit->is_triangle()));
                                         candidate.commit();
                                     })
                                 .trace();
        UInt any_committed_user_id = ~0u;
        $if (!any_committed->miss()) {
            any_committed_user_id = accel.instance_user_id(any_committed->inst);
        };
        result.any_committed = make_uint4(
            any_committed->inst, any_committed->prim,
            any_committed_user_id,
            cast<uint>(any_committed->is_triangle()));
        result.any_callback = make_uint4(
            any_callback_count, any_callback_instance,
            any_callback_primitive, any_callback_user_id);
        result.any_committed_detail = make_float4(
            any_committed->distance(), any_committed->bary.x,
            any_committed->bary.y,
            cast<float>(any_committed->is_triangle()));
        result.any_callback_detail = any_callback_detail;
        return result;
    };

    // This kernel deliberately contains no motion-trace operation or reachable
    // motion callable. Static traversal must therefore evaluate every nested
    // motion level at the API's implicit time zero, including gfx12's flat path.
    Kernel1D trace_static = [&evaluate_static_paths](
                                AccelVar accel, BufferFloat2 probes,
                                BufferVar<StaticMotionProbeResult> output) noexcept {
        auto index = dispatch_id().x;
        auto probe = probes.read(index);
        auto ray = make_ray(make_float3(probe, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);
        output.write(index, evaluate_static_paths(accel, ray));
    };

    const std::array probes{
        make_float2(5.0f, 0.0f),  // MATRIX midpoint at implicit t=0
        make_float2(3.0f, 0.0f),  // MATRIX start pose must be absent at t=0
        make_float2(10.0f, 3.0f), // SRT midpoint at implicit t=0
        make_float2(10.0f, 1.0f), // SRT start pose must be absent at t=0
        make_float2(15.0f, 6.0f), // both nested motion levels at t=0
        make_float2(12.0f, 6.0f), // deforming-child start pose only
        make_float2(15.0f, 4.0f)};// motion-instance start pose only
    const std::array expected{
        ExpectedProbe{"MATRIX implicit-zero hit", true,
                      matrix_instance, matrix_user_id},
        ExpectedProbe{"MATRIX start-pose rejection", false, ~0u, ~0u},
        ExpectedProbe{"SRT implicit-zero hit", true,
                      srt_instance, srt_user_id},
        ExpectedProbe{"SRT start-pose rejection", false, ~0u, ~0u},
        ExpectedProbe{"nested deforming implicit-zero hit", true,
                      nested_instance, nested_user_id},
        ExpectedProbe{"nested deforming child-start rejection", false, ~0u, ~0u},
        ExpectedProbe{"nested deforming instance-start rejection", false, ~0u, ~0u}};

    auto shader = device.compile(trace_static);
    auto probe_buffer = device.create_buffer<float2>(probes.size());
    auto output_buffer =
        device.create_buffer<StaticMotionProbeResult>(probes.size());
    std::array<StaticMotionProbeResult, probes.size()> results{};
    stream << probe_buffer.copy_from(luisa::span{probes})
           << shader(accel, probe_buffer, output_buffer).dispatch(probes.size())
           << output_buffer.copy_to(luisa::span{results})
           << synchronize();

    for (auto i = 0u; i < expected.size(); i++) {
        check_result(expected[i], results[i]);
    }

    // Compile this kernel separately from trace_static. The explicit motion
    // query forces gfx12 to use the generic traversal ABI for the whole module;
    // the static operations must still use implicit time zero rather than the
    // explicit time of the neighboring query.
    Kernel1D trace_generic_forced = [&evaluate_static_paths](
                                        AccelVar accel,
                                        BufferVar<StaticMotionProbeResult> static_output,
                                        BufferUInt4 motion_summary,
                                        BufferFloat4 motion_detail) noexcept {
        auto static_ray = make_ray(make_float3(5.0f, 0.0f, 1.0f),
                                   make_float3(0.0f, 0.0f, -1.0f),
                                   0.0f, 2.0f);
        static_output.write(0u, evaluate_static_paths(accel, static_ray));

        // At t=2, the MATRIX instance is translated by +2 locally and +5 by
        // its outer TLAS transform, so this ray targets its exact endpoint.
        auto motion_ray = make_ray(make_float3(7.0f, 0.0f, 1.0f),
                                   make_float3(0.0f, 0.0f, -1.0f),
                                   0.0f, 2.0f);
        UInt callback_count = 0u;
        auto motion_committed = accel.traverse_motion(motion_ray, 2.0f, {})
                                    .on_surface_candidate(
                                        [&](SurfaceCandidate &candidate) noexcept {
                                            callback_count += 1u;
                                            candidate.commit();
                                        })
                                    .trace();
        UInt user_id = ~0u;
        $if (!motion_committed->miss()) {
            user_id = accel.instance_user_id(motion_committed->inst);
        };
        motion_summary.write(
            0u, make_uint4(motion_committed->inst,
                           motion_committed->prim,
                           callback_count, user_id));
        motion_detail.write(
            0u, make_float4(motion_committed->distance(),
                            motion_committed->bary.x,
                            motion_committed->bary.y,
                            cast<float>(motion_committed->is_triangle())));
    };

    auto generic_forced_shader = device.compile(trace_generic_forced);
    auto generic_static_output =
        device.create_buffer<StaticMotionProbeResult>(1u);
    auto generic_motion_summary = device.create_buffer<uint4>(1u);
    auto generic_motion_detail = device.create_buffer<float4>(1u);
    StaticMotionProbeResult generic_static_result{};
    uint4 generic_motion_result{};
    float4 generic_motion_hit_detail{};
    stream << generic_forced_shader(accel, generic_static_output,
                                    generic_motion_summary,
                                    generic_motion_detail)
                  .dispatch(1u)
           << generic_static_output.copy_to(
                  luisa::span{&generic_static_result, 1u})
           << generic_motion_summary.copy_to(
                  luisa::span{&generic_motion_result, 1u})
           << generic_motion_detail.copy_to(
                  luisa::span{&generic_motion_hit_detail, 1u})
           << synchronize();

    check_result(ExpectedProbe{"generic-path MATRIX implicit-zero hit", true,
                               matrix_instance, matrix_user_id},
                 generic_static_result);
    expect(generic_motion_result.x == matrix_instance)
        << luisa::format("generic-forcing motion query instance: got {}, expected {}",
                         generic_motion_result.x, matrix_instance);
    expect(generic_motion_result.y == kPrimitiveIndex)
        << luisa::format("generic-forcing motion query primitive: got {}, expected {}",
                         generic_motion_result.y, kPrimitiveIndex);
    expect(generic_motion_result.z == 1u)
        << luisa::format("generic-forcing motion query callback count: got {}, expected 1",
                         generic_motion_result.z);
    expect(generic_motion_result.w == matrix_user_id)
        << luisa::format("generic-forcing motion query user ID: got {}, expected {}",
                         generic_motion_result.w, matrix_user_id);
    check_hit_detail("generic-forcing motion query", "MATRIX endpoint",
                     generic_motion_hit_detail);
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP static traversal uses implicit time zero for motion instances"_test = [&] {
        test_hip_static_motion_instance(dc->device);
    };
}
