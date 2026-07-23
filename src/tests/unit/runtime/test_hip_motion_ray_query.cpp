// Test for HIP XIR motion ray-query code generation.
// This test covers:
// - ALL and ANY motion queries through nested MATRIX motion scenes
// - triangle, curve, and procedural candidate callbacks
// - dynamic ray time, outer instance/user IDs, and committed hit semantics
// - callback reference captures and world-ray t-max mutation
// - a static ray query in the same reachable callable as motion queries
// - AOT persistence of the dynamic/global traversal-stack kernel ABI

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct MotionQueryResult {
    uint4 committed;
    uint4 callback;
    float4 committed_detail;
    float4 callback_detail;
};
LUISA_STRUCT(MotionQueryResult, committed, callback,
             committed_detail, callback_detail) {};

namespace {

constexpr auto triangle_instance = 1u;
constexpr auto curve_instance = 2u;
constexpr auto procedural_instance = 3u;
constexpr auto static_instance = 0u;
constexpr auto triangle_user_id = 101u;
constexpr auto curve_user_id = 102u;
constexpr auto procedural_user_id = 103u;
constexpr auto static_user_id = 77u;
constexpr auto tolerance = 2.0e-4f;

struct ExpectedResult {
    uint hit_type;
    uint instance;
    uint primitive;
    uint user_id;
    uint callback_mask;
    uint callback_count;
    uint callback_instance;
    uint callback_user_id;
    float distance;
    float bary_u;
    float bary_v;
    float classification;
    float callback_score;
    float candidate_distance;
    float candidate_tmax_after;
};

void expect_near(luisa::string_view query, size_t index,
                 luisa::string_view field, float actual, float expected) {
    expect(std::abs(actual - expected) < tolerance)
        << luisa::format("{} case {} {}: got {}, expected {}",
                         query, index, field, actual, expected);
}

void check_results(luisa::string_view query,
                   luisa::span<const MotionQueryResult> actual,
                   luisa::span<const ExpectedResult> expected) {
    expect(actual.size() == expected.size());
    for (auto i = 0u; i < expected.size(); i++) {
        auto result = actual[i];
        auto exp = expected[i];
        expect(result.committed.x == exp.hit_type)
            << luisa::format("{} case {} hit type: got {}, expected {}",
                             query, i, result.committed.x, exp.hit_type);
        expect(result.committed.y == exp.instance)
            << luisa::format("{} case {} committed instance: got {}, expected {}",
                             query, i, result.committed.y, exp.instance);
        expect(result.committed.z == exp.primitive)
            << luisa::format("{} case {} committed primitive: got {}, expected {}",
                             query, i, result.committed.z, exp.primitive);
        expect(result.committed.w == exp.user_id)
            << luisa::format("{} case {} committed user ID: got {}, expected {}",
                             query, i, result.committed.w, exp.user_id);
        expect(result.callback.x == exp.callback_mask)
            << luisa::format("{} case {} callback mask: got {}, expected {}",
                             query, i, result.callback.x, exp.callback_mask);
        expect(result.callback.y == exp.callback_count)
            << luisa::format("{} case {} callback count: got {}, expected {}",
                             query, i, result.callback.y, exp.callback_count);
        expect(result.callback.z == exp.callback_instance)
            << luisa::format("{} case {} callback instance: got {}, expected {}",
                             query, i, result.callback.z, exp.callback_instance);
        expect(result.callback.w == exp.callback_user_id)
            << luisa::format("{} case {} callback user ID: got {}, expected {}",
                             query, i, result.callback.w, exp.callback_user_id);
        expect_near(query, i, "committed distance",
                    result.committed_detail.x, exp.distance);
        if (exp.classification != 3.0f) {
            expect_near(query, i, "committed barycentric/curve u",
                        result.committed_detail.y, exp.bary_u);
            expect_near(query, i, "committed barycentric/curve marker",
                        result.committed_detail.z, exp.bary_v);
        }
        expect_near(query, i, "committed classification",
                    result.committed_detail.w, exp.classification);
        expect_near(query, i, "callback score",
                    result.callback_detail.x, exp.callback_score);
        expect_near(query, i, "candidate distance",
                    result.callback_detail.y, exp.candidate_distance);
        expect_near(query, i, "post-commit ray t-max",
                    result.callback_detail.z, exp.candidate_tmax_after);
        expect_near(query, i, "candidate classification",
                    result.callback_detail.w, exp.classification);
    }
}

void test_hip_motion_ray_query(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO("Skipping HIP-specific motion ray-query test on backend '{}'.",
                   device.backend_name());
        return;
    }

    log_level_verbose();

    // Every moving geometry starts at x=-2 and ends at x=+2. The probes use
    // three different, non-endpoint times, so dropping the time operand makes
    // every motion query miss. An outer translation then places the probes at
    // y=0/2/4 and makes the reported TLAS instance observable.
    const std::array vertices{
        make_float3(-0.5f, -0.5f, 0.0f),
        make_float3(0.5f, -0.5f, 0.0f),
        make_float3(0.0f, 0.5f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    const std::array control_points{
        make_float4(-0.5f, 0.0f, 0.0f, 0.1f),
        make_float4(0.5f, 0.0f, 0.0f, 0.1f)};
    const std::array segments{0u};
    const std::array aabbs{
        AABB{.packed_min = {-0.5f, -0.5f, -0.1f},
             .packed_max = {0.5f, 0.5f, 0.1f}}};

    auto stream = device.create_stream();
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto control_point_buffer = device.create_buffer<float4>(control_points.size());
    auto segment_buffer = device.create_buffer<uint>(segments.size());
    auto aabb_buffer = device.create_buffer<AABB>(aabbs.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto curve = device.create_curve(
        CurveBasis::PIECEWISE_LINEAR, control_point_buffer, segment_buffer);
    auto procedural = device.create_procedural_primitive(aabb_buffer);

    AccelMotionOption motion_option{};
    motion_option.keyframe_count = 2u;
    motion_option.time_start = 0.0f;
    motion_option.time_end = 1.0f;
    motion_option.mode = AccelMotionMode::MATRIX;
    auto moving_triangle = device.create_motion_instance(mesh, motion_option);
    auto moving_curve = device.create_motion_instance(curve, motion_option);
    auto moving_procedural = device.create_motion_instance(procedural, motion_option);
    const std::array keyframes{
        translation(-2.0f, 0.0f, 0.0f),
        translation(2.0f, 0.0f, 0.0f)};
    moving_triangle.set_keyframes(luisa::span{keyframes});
    moving_curve.set_keyframes(luisa::span{keyframes});
    moving_procedural.set_keyframes(luisa::span{keyframes});

    auto motion_accel = device.create_accel();
    // A far-away decoy reserves outer instance zero, proving that nested
    // private-instance traversal returns the outer TLAS index rather than the
    // motion scene's sole child index.
    motion_accel.emplace_back(mesh, translation(-100.0f, 0.0f, 0.0f),
                              0xffu, true, 1u);
    motion_accel.emplace_back(moving_triangle, translation(5.0f, 0.0f, 0.0f),
                              0xffu, false, triangle_user_id);
    motion_accel.emplace_back(moving_curve, translation(5.0f, 2.0f, 0.0f),
                              0xffu, false, curve_user_id);
    motion_accel.emplace_back(moving_procedural, translation(5.0f, 4.0f, 0.0f),
                              0xffu, false, procedural_user_id);

    // This separate static scene is deliberately queried by traverse(), not a
    // direct trace. On gfx12 the containing module must route this static query
    // and both motion-query operations through a compatible generic-stack ABI.
    auto static_accel = device.create_accel();
    static_accel.emplace_back(mesh, translation(20.0f, 0.0f, 0.0f),
                              0xffu, false, static_user_id);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << control_point_buffer.copy_from(luisa::span{control_points})
           << segment_buffer.copy_from(luisa::span{segments})
           << aabb_buffer.copy_from(luisa::span{aabbs})
           << mesh.build()
           << curve.build()
           << procedural.build()
           << moving_triangle.build()
           << moving_curve.build()
           << moving_procedural.build()
           << motion_accel.build()
           << static_accel.build()
           << synchronize();

    Callable trace_queries = [](
                                 AccelVar motion_accel,
                                 AccelVar static_accel,
                                 UInt index,
                                 BufferVar<MotionQueryResult> all_output,
                                 BufferVar<MotionQueryResult> any_output,
                                 BufferVar<MotionQueryResult> static_output) noexcept {
        auto time = ite(index == 0u, 0.25f,
                        ite(index == 1u, 0.5f, 0.75f));
        auto origin_x = ite(index == 0u, 4.0f,
                            ite(index == 1u, 5.0f, 6.0f));
        auto origin_y = cast<float>(index) * 2.0f;
        auto ray = make_ray(make_float3(origin_x, origin_y, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);
        auto options = AccelTraceOptions{
            .curve_bases = {CurveBasis::PIECEWISE_LINEAR}};

        UInt all_callback_mask = 0u;
        UInt all_callback_count = 0u;
        UInt all_callback_instance = ~0u;
        UInt all_callback_user_id = ~0u;
        UInt all_score = 100u + index;
        Float all_candidate_distance = -1.0f;
        Float all_candidate_tmax_after = -1.0f;
        Float all_candidate_classification = 0.0f;
        auto all_committed = motion_accel.traverse_motion(ray, time, options)
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         auto kind = ite(hit->is_curve(), 2u, 1u);
                                         all_callback_mask = all_callback_mask | kind;
                                         all_callback_count += 1u;
                                         all_callback_instance = hit->inst;
                                         all_callback_user_id = motion_accel.instance_user_id(hit->inst);
                                         all_score += kind * 10u;
                                         all_candidate_distance = hit->distance();
                                         all_candidate_classification = cast<float>(kind);
                                         candidate.commit();
                                         all_candidate_tmax_after = candidate.ray()->t_max();
                                     })
                                 .on_procedural_candidate(
                                     [&](ProceduralCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         all_callback_mask = all_callback_mask | 4u;
                                         all_callback_count += 1u;
                                         all_callback_instance = hit->inst;
                                         all_callback_user_id = motion_accel.instance_user_id(hit->inst);
                                         all_score += 30u;
                                         all_candidate_distance = 1.0f;
                                         all_candidate_classification = 3.0f;
                                         candidate.commit(1.0f);
                                         all_candidate_tmax_after = candidate.ray()->t_max();
                                     })
                                 .trace();
        UInt all_user_id = ~0u;
        $if (!all_committed->miss()) {
            all_user_id = motion_accel.instance_user_id(all_committed->inst);
        };
        UInt all_classification = ite(
            all_committed->is_triangle(), 1u,
            ite(all_committed->is_curve(), 2u,
                ite(all_committed->is_procedural(), 3u, 0u)));
        Var<MotionQueryResult> all_result;
        all_result.committed = make_uint4(
            all_committed->hit_type, all_committed->inst,
            all_committed->prim, all_user_id);
        all_result.callback = make_uint4(
            all_callback_mask, all_callback_count,
            all_callback_instance, all_callback_user_id);
        all_result.committed_detail = make_float4(
            all_committed->distance(), all_committed->bary.x,
            all_committed->bary.y, cast<float>(all_classification));
        all_result.callback_detail = make_float4(
            cast<float>(all_score), all_candidate_distance,
            all_candidate_tmax_after, all_candidate_classification);
        all_output.write(index, all_result);

        UInt any_callback_mask = 0u;
        UInt any_callback_count = 0u;
        UInt any_callback_instance = ~0u;
        UInt any_callback_user_id = ~0u;
        UInt any_score = 200u + index;
        Float any_candidate_distance = -1.0f;
        Float any_candidate_tmax_after = -1.0f;
        Float any_candidate_classification = 0.0f;
        auto any_committed = motion_accel.traverse_any_motion(ray, time, options)
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         auto kind = ite(hit->is_curve(), 2u, 1u);
                                         any_callback_mask = any_callback_mask | kind;
                                         any_callback_count += 1u;
                                         any_callback_instance = hit->inst;
                                         any_callback_user_id = motion_accel.instance_user_id(hit->inst);
                                         any_score += kind * 10u;
                                         any_candidate_distance = hit->distance();
                                         any_candidate_classification = cast<float>(kind);
                                         candidate.commit();
                                         any_candidate_tmax_after = candidate.ray()->t_max();
                                     })
                                 .on_procedural_candidate(
                                     [&](ProceduralCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         any_callback_mask = any_callback_mask | 4u;
                                         any_callback_count += 1u;
                                         any_callback_instance = hit->inst;
                                         any_callback_user_id = motion_accel.instance_user_id(hit->inst);
                                         any_score += 30u;
                                         any_candidate_distance = 1.0f;
                                         any_candidate_classification = 3.0f;
                                         candidate.commit(1.0f);
                                         any_candidate_tmax_after = candidate.ray()->t_max();
                                     })
                                 .trace();
        UInt any_user_id = ~0u;
        $if (!any_committed->miss()) {
            any_user_id = motion_accel.instance_user_id(any_committed->inst);
        };
        UInt any_classification = ite(
            any_committed->is_triangle(), 1u,
            ite(any_committed->is_curve(), 2u,
                ite(any_committed->is_procedural(), 3u, 0u)));
        Var<MotionQueryResult> any_result;
        any_result.committed = make_uint4(
            any_committed->hit_type, any_committed->inst,
            any_committed->prim, any_user_id);
        any_result.callback = make_uint4(
            any_callback_mask, any_callback_count,
            any_callback_instance, any_callback_user_id);
        any_result.committed_detail = make_float4(
            any_committed->distance(), any_committed->bary.x,
            any_committed->bary.y, cast<float>(any_classification));
        any_result.callback_detail = make_float4(
            cast<float>(any_score), any_candidate_distance,
            any_candidate_tmax_after, any_candidate_classification);
        any_output.write(index, any_result);

        auto static_ray = make_ray(make_float3(20.0f, 0.0f, 1.0f),
                                   make_float3(0.0f, 0.0f, -1.0f),
                                   0.0f, 2.0f);
        UInt static_callback_mask = 0u;
        UInt static_callback_count = 0u;
        UInt static_callback_instance = ~0u;
        UInt static_callback_user_id = ~0u;
        UInt static_score = 40u + index;
        Float static_candidate_distance = -1.0f;
        Float static_candidate_tmax_after = -1.0f;
        auto static_committed = static_accel.traverse(static_ray, {})
                                    .on_surface_candidate(
                                        [&](SurfaceCandidate &candidate) noexcept {
                                            auto hit = candidate.hit();
                                            static_callback_mask = static_callback_mask | 1u;
                                            static_callback_count += 1u;
                                            static_callback_instance = hit->inst;
                                            static_callback_user_id = static_accel.instance_user_id(hit->inst);
                                            static_score += 7u;
                                            static_candidate_distance = hit->distance();
                                            candidate.commit();
                                            static_candidate_tmax_after = candidate.ray()->t_max();
                                        })
                                    .trace();
        UInt committed_static_user_id = ~0u;
        $if (!static_committed->miss()) {
            committed_static_user_id = static_accel.instance_user_id(static_committed->inst);
        };
        Var<MotionQueryResult> static_result;
        static_result.committed = make_uint4(
            static_committed->hit_type, static_committed->inst,
            static_committed->prim, committed_static_user_id);
        static_result.callback = make_uint4(
            static_callback_mask, static_callback_count,
            static_callback_instance, static_callback_user_id);
        static_result.committed_detail = make_float4(
            static_committed->distance(), static_committed->bary.x,
            static_committed->bary.y,
            cast<float>(static_committed->is_triangle()));
        static_result.callback_detail = make_float4(
            cast<float>(static_score), static_candidate_distance,
            static_candidate_tmax_after, 1.0f);
        static_output.write(index, static_result);
    };
    trace_queries.function_builder()->set_name(
        "hip_mixed_static_and_motion_ray_query_callable");

    Kernel1D trace = [&trace_queries](
                         AccelVar motion_accel,
                         AccelVar static_accel,
                         BufferVar<MotionQueryResult> all_output,
                         BufferVar<MotionQueryResult> any_output,
                         BufferVar<MotionQueryResult> static_output) noexcept {
        auto index = dispatch_id().x;
        trace_queries(motion_accel, static_accel, index,
                      all_output, any_output, static_output);
    };

    constexpr auto case_count = 3u;
    auto all_output = device.create_buffer<MotionQueryResult>(case_count);
    auto any_output = device.create_buffer<MotionQueryResult>(case_count);
    auto static_output = device.create_buffer<MotionQueryResult>(case_count);
    auto package_path = std::filesystem::absolute(
        "test_hip_motion_ray_query_aot.bytes");
    auto package_name = luisa::string{package_path.string()};
    std::error_code package_ec;
    std::filesystem::remove(package_path, package_ec);
    {
        ShaderOption option{
            .compile_only = true,
            .name = package_name};
        [[maybe_unused]] auto compiled = device.compile(trace, option);
    }
    expect(std::filesystem::is_regular_file(package_path))
        << "HIP motion ray-query AOT package was not written";
    auto shader = device.load_shader<
        1, Accel, Accel,
        Buffer<MotionQueryResult>, Buffer<MotionQueryResult>,
        Buffer<MotionQueryResult>>(package_name);
    std::array<MotionQueryResult, case_count> host_all{};
    std::array<MotionQueryResult, case_count> host_any{};
    std::array<MotionQueryResult, case_count> host_static{};
    stream << shader(motion_accel, static_accel,
                     all_output, any_output, static_output)
                  .dispatch(case_count)
           << all_output.copy_to(luisa::span{host_all})
           << any_output.copy_to(luisa::span{host_any})
           << static_output.copy_to(luisa::span{host_static})
           << synchronize();

    constexpr auto surface = static_cast<uint>(HitType::Surface);
    constexpr auto procedural_hit = static_cast<uint>(HitType::Procedural);
    const std::array expected_all{
        ExpectedResult{surface, triangle_instance, 0u, triangle_user_id,
                       1u, 1u, triangle_instance, triangle_user_id,
                       1.0f, 0.25f, 0.5f, 1.0f,
                       110.0f, 1.0f, 1.0f},
        ExpectedResult{surface, curve_instance, 0u, curve_user_id,
                       2u, 1u, curve_instance, curve_user_id,
                       0.9f, 0.5f, -1.0f, 2.0f,
                       121.0f, 0.9f, 0.9f},
        ExpectedResult{procedural_hit, procedural_instance, 0u, procedural_user_id,
                       4u, 1u, procedural_instance, procedural_user_id,
                       1.0f, 0.0f, 0.0f, 3.0f,
                       132.0f, 1.0f, 1.0f}};
    check_results("motion ALL", luisa::span{host_all},
                  luisa::span{expected_all});

    const std::array expected_any{
        ExpectedResult{surface, triangle_instance, 0u, triangle_user_id,
                       1u, 1u, triangle_instance, triangle_user_id,
                       1.0f, 0.25f, 0.5f, 1.0f,
                       210.0f, 1.0f, 1.0f},
        ExpectedResult{surface, curve_instance, 0u, curve_user_id,
                       2u, 1u, curve_instance, curve_user_id,
                       0.9f, 0.5f, -1.0f, 2.0f,
                       221.0f, 0.9f, 0.9f},
        ExpectedResult{procedural_hit, procedural_instance, 0u, procedural_user_id,
                       4u, 1u, procedural_instance, procedural_user_id,
                       1.0f, 0.0f, 0.0f, 3.0f,
                       232.0f, 1.0f, 1.0f}};
    check_results("motion ANY", luisa::span{host_any},
                  luisa::span{expected_any});

    const std::array expected_static{
        ExpectedResult{surface, static_instance, 0u, static_user_id,
                       1u, 1u, static_instance, static_user_id,
                       1.0f, 0.25f, 0.5f, 1.0f,
                       47.0f, 1.0f, 1.0f},
        ExpectedResult{surface, static_instance, 0u, static_user_id,
                       1u, 1u, static_instance, static_user_id,
                       1.0f, 0.25f, 0.5f, 1.0f,
                       48.0f, 1.0f, 1.0f},
        ExpectedResult{surface, static_instance, 0u, static_user_id,
                       1u, 1u, static_instance, static_user_id,
                       1.0f, 0.25f, 0.5f, 1.0f,
                       49.0f, 1.0f, 1.0f}};
    check_results("mixed static", luisa::span{host_static},
                  luisa::span{expected_static});

    std::filesystem::remove(package_path, package_ec);
    expect(!std::filesystem::exists(package_path))
        << "HIP motion ray-query AOT package cleanup failed";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP XIR motion ray queries preserve nested candidate semantics"_test = [&] {
        test_hip_motion_ray_query(dc->device);
    };
}
