// Exact HIP coverage for deforming mesh geometry motion.
// This test covers:
// - three stacked vertex keyframes over a non-default shutter interval
// - endpoint and piecewise-linear interior interpolation
// - exact triangle rejection after a conservative swept-AABB candidate
// - direct closest/any motion traces and ALL/ANY motion ray queries
// - separately compiled static-only closest/any and ALL/ANY query paths
// - independent start-vanish and end-vanish boundary semantics
// - owned build snapshots after source-buffer overwrite and destruction
// - 16-byte vertex stride and nonzero vertex/triangle buffer-view offsets
// - repeated allow-update rebuilds with changed motion keyframes
// - rejection of unreferenced non-finite vertices and invalid triangle indices

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>
#include <cerrno>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <utility>

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
#include <sys/wait.h>
#include <unistd.h>
#define LUISA_TEST_HIP_MOTION_MESH_HAS_EXPECTED_FAILURE_SUBPROCESS 1
#else
#define LUISA_TEST_HIP_MOTION_MESH_HAS_EXPECTED_FAILURE_SUBPROCESS 0
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct ExpectedProbe {
    luisa::string_view name;
    bool hit;
    uint instance;
};

constexpr auto primitive_index = 0u;
constexpr auto expected_distance = 1.0f;
constexpr auto expected_barycentric = make_float2(0.25f, 0.5f);
constexpr auto float_tolerance = 2.0e-4f;
constexpr auto invalid_non_finite_mode = "--invalid-unreferenced-non-finite";
constexpr auto invalid_triangle_index_mode = "--invalid-triangle-index";

void expect_near(luisa::string_view path, luisa::string_view probe,
                 luisa::string_view field, float actual, float expected) {
    expect(std::abs(actual - expected) < float_tolerance)
        << luisa::format("{} probe '{}' {}: got {}, expected {}",
                         path, probe, field, actual, expected);
}

void check_trace_result(luisa::string_view probe,
                        const ExpectedProbe &expected,
                        uint4 summary, float4 detail) {
    if (expected.hit) {
        expect(summary.x == expected.instance)
            << luisa::format("direct probe '{}' instance: got {}, expected {}",
                             probe, summary.x, expected.instance);
        expect(summary.y == primitive_index)
            << luisa::format("direct probe '{}' primitive: got {}, expected {}",
                             probe, summary.y, primitive_index);
        expect(summary.z == 1u)
            << luisa::format("direct probe '{}' any-hit trace missed", probe);
        expect(summary.w == 1u)
            << luisa::format("direct probe '{}' was not classified as a triangle",
                             probe);
        expect_near("direct", probe, "distance", detail.x,
                    expected_distance);
        expect_near("direct", probe, "barycentric x", detail.y,
                    expected_barycentric.x);
        expect_near("direct", probe, "barycentric y", detail.z,
                    expected_barycentric.y);
    } else {
        expect(summary.x == ~0u)
            << luisa::format("direct probe '{}' unexpectedly hit instance {}",
                             probe, summary.x);
        expect(summary.z == 0u)
            << luisa::format("direct probe '{}' any-hit trace unexpectedly hit",
                             probe);
        expect(summary.w == 0u)
            << luisa::format("direct probe '{}' miss was classified as a triangle",
                             probe);
    }
}

void check_query_result(luisa::string_view path, luisa::string_view probe,
                        const ExpectedProbe &expected,
                        uint4 summary, float4 detail) {
    if (expected.hit) {
        expect(summary.x == expected.instance)
            << luisa::format("{} probe '{}' instance: got {}, expected {}",
                             path, probe, summary.x, expected.instance);
        expect(summary.y == primitive_index)
            << luisa::format("{} probe '{}' primitive: got {}, expected {}",
                             path, probe, summary.y, primitive_index);
        expect(summary.z == 1u)
            << luisa::format("{} probe '{}' callback count: got {}, expected 1",
                             path, probe, summary.z);
        expect(summary.w == expected.instance)
            << luisa::format("{} probe '{}' callback instance: got {}, expected {}",
                             path, probe, summary.w, expected.instance);
        expect_near(path, probe, "distance", detail.x, expected_distance);
        expect_near(path, probe, "barycentric x", detail.y,
                    expected_barycentric.x);
        expect_near(path, probe, "barycentric y", detail.z,
                    expected_barycentric.y);
        expect_near(path, probe, "triangle classification", detail.w, 1.0f);
    } else {
        expect(summary.x == ~0u)
            << luisa::format("{} probe '{}' unexpectedly committed instance {}",
                             path, probe, summary.x);
        expect(summary.z == 0u)
            << luisa::format("{} probe '{}' reached {} surface callbacks; "
                             "the exact intersection should reject its swept-AABB candidate",
                             path, probe, summary.z);
        expect(summary.w == ~0u)
            << luisa::format("{} probe '{}' unexpectedly reported callback instance {}",
                             path, probe, summary.w);
        expect_near(path, probe, "miss classification", detail.w, 0.0f);
    }
}

[[nodiscard]] auto make_padded_motion_vertices(
    const std::array<float2, 3u> &centers) noexcept {
    std::array<float4, 9u> vertices{};
    for (auto keyframe = 0u; keyframe < centers.size(); keyframe++) {
        auto center = centers[keyframe];
        auto base = keyframe * 3u;
        auto padding = 1000.0f + static_cast<float>(keyframe);
        vertices[base + 0u] = make_float4(
            center.x - 0.5f, center.y - 0.5f, -0.05f, padding);
        vertices[base + 1u] = make_float4(
            center.x + 0.5f, center.y - 0.5f, -0.05f, padding);
        vertices[base + 2u] = make_float4(
            center.x, center.y + 0.5f, 0.05f, padding);
    }
    return vertices;
}

[[nodiscard]] auto make_update_probes(
    const std::array<float2, 3u> &centers) noexcept {
    auto midpoint = [](float2 lhs, float2 rhs) noexcept {
        return (lhs + rhs) * 0.5f;
    };
    auto first_midpoint = midpoint(centers[0], centers[1]);
    auto second_midpoint = midpoint(centers[1], centers[2]);
    return std::array{
        make_float4(centers[0].x, centers[0].y, 2.0f, 0.0f),
        make_float4(first_midpoint.x, first_midpoint.y, 3.0f, 0.0f),
        make_float4(centers[1].x, centers[1].y, 4.0f, 0.0f),
        make_float4(second_midpoint.x, second_midpoint.y, 5.0f, 0.0f),
        make_float4(centers[2].x, centers[2].y, 6.0f, 0.0f)};
}

void test_hip_motion_mesh(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO("Skipping HIP-specific deforming motion-mesh test on backend '{}'.",
                   device.backend_name());
        return;
    }

    log_level_verbose();

    // Triangle indices are local to each keyframe. The keyframes deliberately
    // change both position and shape, so a passing result requires per-vertex
    // interpolation rather than an instance-translation approximation.
    // Every positive probe targets barycentrics (u, v) = (0.25, 0.5).
    const std::array vertices{
        // keyframe 0: target point (-2, 0)
        make_float3(-2.5f, -0.5f, -0.05f),
        make_float3(-1.5f, -0.5f, -0.05f),
        make_float3(-2.0f, 0.5f, 0.05f),
        // keyframe 1: target point (0, 0)
        make_float3(-1.0f, -1.0f, -0.05f),
        make_float3(1.0f, -1.0f, -0.05f),
        make_float3(0.0f, 1.0f, 0.05f),
        // keyframe 2: target point (2, 0.25)
        make_float3(1.75f, -0.25f, -0.05f),
        make_float3(2.25f, -0.25f, -0.05f),
        make_float3(2.0f, 0.75f, 0.05f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};

    auto stream = device.create_stream();
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());

    auto make_motion_option = [](bool vanish_start, bool vanish_end) noexcept {
        AccelOption option{};
        option.motion.keyframe_count = 3u;
        option.motion.time_start = 2.0f;
        option.motion.time_end = 6.0f;
        option.motion.should_vanish_start = vanish_start;
        option.motion.should_vanish_end = vanish_end;
        return option;
    };
    auto clamped_mesh = device.create_mesh(
        vertex_buffer, triangle_buffer, make_motion_option(false, false));
    auto vanish_start_mesh = device.create_mesh(
        vertex_buffer, triangle_buffer, make_motion_option(true, false));
    auto vanish_end_mesh = device.create_mesh(
        vertex_buffer, triangle_buffer, make_motion_option(false, true));

    auto accel = device.create_accel();
    accel.emplace_back(clamped_mesh, make_float4x4(1.0f),
                       0xffu, false, 100u);
    accel.emplace_back(vanish_start_mesh, translation(0.0f, 4.0f, 0.0f),
                       0xffu, false, 101u);
    accel.emplace_back(vanish_end_mesh, translation(0.0f, 8.0f, 0.0f),
                       0xffu, false, 102u);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << clamped_mesh.build()
           << vanish_start_mesh.build()
           << vanish_end_mesh.build()
           << accel.build()
           << synchronize();

    // A built motion mesh must own the geometry snapshot consumed during
    // traversal. Overwrite both sources with independently destructive data,
    // wait for those writes, and then release the allocations. Every trace
    // below happens afterward and without rebuilding any mesh or the TLAS.
    const std::array overwritten_vertices{
        make_float3(100.0f, 100.0f, 100.0f),
        make_float3(101.0f, 100.0f, 100.0f),
        make_float3(100.0f, 101.0f, 100.0f),
        make_float3(110.0f, 110.0f, 110.0f),
        make_float3(111.0f, 110.0f, 110.0f),
        make_float3(110.0f, 111.0f, 110.0f),
        make_float3(120.0f, 120.0f, 120.0f),
        make_float3(121.0f, 120.0f, 120.0f),
        make_float3(120.0f, 121.0f, 120.0f)};
    const std::array overwritten_triangles{Triangle{0u, 0u, 0u}};
    stream << vertex_buffer.copy_from(luisa::span{overwritten_vertices})
           << triangle_buffer.copy_from(luisa::span{overwritten_triangles})
           << synchronize();
    vertex_buffer = {};
    triangle_buffer = {};
    expect(!static_cast<bool>(vertex_buffer))
        << "motion-mesh source vertex buffer was not released";
    expect(!static_cast<bool>(triangle_buffer))
        << "motion-mesh source triangle buffer was not released";

    // Compile this kernel independently. It intentionally contains no motion
    // operation or reachable motion callable, so gfx12 must select its flat
    // static traversal wrappers while still intersecting motion-mesh custom
    // leaves at the implicit time zero.
    Kernel1D trace_static = [](AccelVar accel, BufferFloat4 probes,
                               BufferUInt4 direct_summaries,
                               BufferFloat4 direct_details,
                               BufferUInt4 all_summaries,
                               BufferFloat4 all_details,
                               BufferUInt4 any_summaries,
                               BufferFloat4 any_details) noexcept {
        auto index = dispatch_id().x;
        auto probe = probes.read(index);
        auto ray = make_ray(make_float3(probe.x, probe.y, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);

        auto closest = accel.intersect(ray, {});
        auto any = accel.intersect_any(ray, {});
        direct_summaries.write(
            index, make_uint4(closest->inst, closest->prim,
                              cast<uint>(any),
                              cast<uint>(closest->is_triangle())));
        direct_details.write(
            index, make_float4(closest->distance(), closest->bary.x,
                               closest->bary.y,
                               cast<float>(closest->is_triangle())));

        UInt all_callback_count = 0u;
        UInt all_callback_instance = ~0u;
        auto all_committed = accel.traverse(ray, {})
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         all_callback_count += 1u;
                                         all_callback_instance = hit->inst;
                                         candidate.commit();
                                     })
                                 .trace();
        all_summaries.write(
            index, make_uint4(all_committed->inst, all_committed->prim,
                              all_callback_count, all_callback_instance));
        all_details.write(
            index, make_float4(all_committed->distance(),
                               all_committed->bary.x,
                               all_committed->bary.y,
                               cast<float>(all_committed->is_triangle())));

        UInt any_callback_count = 0u;
        UInt any_callback_instance = ~0u;
        auto any_committed = accel.traverse_any(ray, {})
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         any_callback_count += 1u;
                                         any_callback_instance = hit->inst;
                                         candidate.commit();
                                     })
                                 .trace();
        any_summaries.write(
            index, make_uint4(any_committed->inst, any_committed->prim,
                              any_callback_count, any_callback_instance));
        any_details.write(
            index, make_float4(any_committed->distance(),
                               any_committed->bary.x,
                               any_committed->bary.y,
                               cast<float>(any_committed->is_triangle())));
    };
    constexpr auto static_case_count = 4u;
    const std::array<float4, static_case_count> static_probes{
        make_float4(-2.0f, 0.0f, 0.0f, 0.0f), // unflagged mesh clamps to start
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),  // swept AABB, exact start miss
        make_float4(-2.0f, 4.0f, 0.0f, 0.0f), // start-vanish mesh is absent
        make_float4(-2.0f, 8.0f, 0.0f, 0.0f)};// end-only mesh clamps to start
    const std::array<ExpectedProbe, static_case_count> static_expected{
        ExpectedProbe{"static implicit-start hit", true, 0u},
        ExpectedProbe{"static swept AABB exact rejection", false, ~0u},
        ExpectedProbe{"static implicit start vanish", false, ~0u},
        ExpectedProbe{"static end-only start clamp", true, 2u}};

    auto static_shader = device.compile(trace_static);
    auto static_probe_buffer =
        device.create_buffer<float4>(static_case_count);
    auto static_direct_summary_buffer =
        device.create_buffer<uint4>(static_case_count);
    auto static_direct_detail_buffer =
        device.create_buffer<float4>(static_case_count);
    auto static_all_summary_buffer =
        device.create_buffer<uint4>(static_case_count);
    auto static_all_detail_buffer =
        device.create_buffer<float4>(static_case_count);
    auto static_any_summary_buffer =
        device.create_buffer<uint4>(static_case_count);
    auto static_any_detail_buffer =
        device.create_buffer<float4>(static_case_count);
    std::array<uint4, static_case_count> static_direct_summaries{};
    std::array<float4, static_case_count> static_direct_details{};
    std::array<uint4, static_case_count> static_all_summaries{};
    std::array<float4, static_case_count> static_all_details{};
    std::array<uint4, static_case_count> static_any_summaries{};
    std::array<float4, static_case_count> static_any_details{};
    stream << static_probe_buffer.copy_from(luisa::span{static_probes})
           << static_shader(accel, static_probe_buffer,
                            static_direct_summary_buffer,
                            static_direct_detail_buffer,
                            static_all_summary_buffer,
                            static_all_detail_buffer,
                            static_any_summary_buffer,
                            static_any_detail_buffer)
                  .dispatch(static_case_count)
           << static_direct_summary_buffer.copy_to(
                  luisa::span{static_direct_summaries})
           << static_direct_detail_buffer.copy_to(
                  luisa::span{static_direct_details})
           << static_all_summary_buffer.copy_to(
                  luisa::span{static_all_summaries})
           << static_all_detail_buffer.copy_to(
                  luisa::span{static_all_details})
           << static_any_summary_buffer.copy_to(
                  luisa::span{static_any_summaries})
           << static_any_detail_buffer.copy_to(
                  luisa::span{static_any_details})
           << synchronize();
    for (auto i = 0u; i < static_case_count; i++) {
        auto &exp = static_expected[i];
        check_trace_result(exp.name, exp,
                           static_direct_summaries[i],
                           static_direct_details[i]);
        check_query_result("static ALL query", exp.name, exp,
                           static_all_summaries[i],
                           static_all_details[i]);
        check_query_result("static ANY query", exp.name, exp,
                           static_any_summaries[i],
                           static_any_details[i]);
    }

    Kernel1D trace_motion = [](AccelVar accel, BufferFloat4 probes,
                               BufferUInt4 direct_summaries,
                               BufferFloat4 direct_details,
                               BufferUInt4 all_summaries,
                               BufferFloat4 all_details,
                               BufferUInt4 any_summaries,
                               BufferFloat4 any_details) noexcept {
        auto index = dispatch_id().x;
        auto probe = probes.read(index);
        auto ray = make_ray(make_float3(probe.x, probe.y, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);

        auto closest = accel.intersect_motion(ray, probe.z, {});
        auto any = accel.intersect_any_motion(ray, probe.z, {});
        direct_summaries.write(
            index, make_uint4(closest->inst, closest->prim,
                              cast<uint>(any),
                              cast<uint>(closest->is_triangle())));
        direct_details.write(
            index, make_float4(closest->distance(), closest->bary.x,
                               closest->bary.y,
                               cast<float>(closest->is_triangle())));

        UInt all_callback_count = 0u;
        UInt all_callback_instance = ~0u;
        auto all_committed = accel.traverse_motion(ray, probe.z, {})
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         all_callback_count += 1u;
                                         all_callback_instance = hit->inst;
                                         candidate.commit();
                                     })
                                 .trace();
        all_summaries.write(
            index, make_uint4(all_committed->inst, all_committed->prim,
                              all_callback_count, all_callback_instance));
        all_details.write(
            index, make_float4(all_committed->distance(),
                               all_committed->bary.x,
                               all_committed->bary.y,
                               cast<float>(all_committed->is_triangle())));

        UInt any_callback_count = 0u;
        UInt any_callback_instance = ~0u;
        auto any_committed = accel.traverse_any_motion(ray, probe.z, {})
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         any_callback_count += 1u;
                                         any_callback_instance = hit->inst;
                                         candidate.commit();
                                     })
                                 .trace();
        any_summaries.write(
            index, make_uint4(any_committed->inst, any_committed->prim,
                              any_callback_count, any_callback_instance));
        any_details.write(
            index, make_float4(any_committed->distance(),
                               any_committed->bary.x,
                               any_committed->bary.y,
                               cast<float>(any_committed->is_triangle())));
    };

    // z stores ray time. The first seven probes cover both interpolation
    // segments, both shutter endpoints, clamping outside the shutter, and an
    // intentional swept-bound false positive. The remaining probes isolate
    // vanish behavior immediately outside and exactly on each boundary.
    constexpr auto case_count = 14u;
    const std::array<float4, case_count> probes{
        make_float4(-2.0f, 0.0f, 2.0f, 0.0f), // keyframe 0
        make_float4(-1.0f, 0.0f, 3.0f, 0.0f), // keyframe 0 -> 1 midpoint
        make_float4(0.0f, 0.0f, 4.0f, 0.0f),  // keyframe 1
        make_float4(1.0f, 0.125f, 5.0f, 0.0f),// keyframe 1 -> 2 midpoint
        make_float4(2.0f, 0.25f, 6.0f, 0.0f), // keyframe 2
        make_float4(-2.0f, 0.0f, 4.0f, 0.0f), // swept AABB, exact miss
        make_float4(-2.0f, 0.0f, 1.0f, 0.0f), // clamp before start
        make_float4(2.0f, 0.25f, 7.0f, 0.0f), // clamp after end
        make_float4(-2.0f, 4.0f, 1.0f, 0.0f), // vanish before start
        make_float4(-2.0f, 4.0f, 2.0f, 0.0f), // visible at start
        make_float4(2.0f, 4.25f, 7.0f, 0.0f), // start-only flag still clamps end
        make_float4(-2.0f, 8.0f, 1.0f, 0.0f), // end-only flag still clamps start
        make_float4(2.0f, 8.25f, 6.0f, 0.0f), // visible at end
        make_float4(2.0f, 8.25f, 7.0f, 0.0f)};// vanish after end
    const std::array<ExpectedProbe, case_count> expected{
        ExpectedProbe{"keyframe 0 endpoint", true, 0u},
        ExpectedProbe{"first segment midpoint", true, 0u},
        ExpectedProbe{"keyframe 1 endpoint", true, 0u},
        ExpectedProbe{"second segment midpoint", true, 0u},
        ExpectedProbe{"keyframe 2 endpoint", true, 0u},
        ExpectedProbe{"swept AABB exact rejection", false, ~0u},
        ExpectedProbe{"unflagged start clamp", true, 0u},
        ExpectedProbe{"unflagged end clamp", true, 0u},
        ExpectedProbe{"start vanish outside", false, ~0u},
        ExpectedProbe{"start vanish boundary", true, 1u},
        ExpectedProbe{"start-only end clamp", true, 1u},
        ExpectedProbe{"end-only start clamp", true, 2u},
        ExpectedProbe{"end vanish boundary", true, 2u},
        ExpectedProbe{"end vanish outside", false, ~0u}};
    auto shader = device.compile(trace_motion);
    auto probe_buffer = device.create_buffer<float4>(probes.size());
    auto direct_summary_buffer = device.create_buffer<uint4>(probes.size());
    auto direct_detail_buffer = device.create_buffer<float4>(probes.size());
    auto all_summary_buffer = device.create_buffer<uint4>(probes.size());
    auto all_detail_buffer = device.create_buffer<float4>(probes.size());
    auto any_summary_buffer = device.create_buffer<uint4>(probes.size());
    auto any_detail_buffer = device.create_buffer<float4>(probes.size());

    std::array<uint4, case_count> direct_summaries{};
    std::array<float4, case_count> direct_details{};
    std::array<uint4, case_count> all_summaries{};
    std::array<float4, case_count> all_details{};
    std::array<uint4, case_count> any_summaries{};
    std::array<float4, case_count> any_details{};
    stream << probe_buffer.copy_from(luisa::span{probes})
           << shader(accel, probe_buffer,
                     direct_summary_buffer, direct_detail_buffer,
                     all_summary_buffer, all_detail_buffer,
                     any_summary_buffer, any_detail_buffer)
                  .dispatch(probes.size())
           << direct_summary_buffer.copy_to(luisa::span{direct_summaries})
           << direct_detail_buffer.copy_to(luisa::span{direct_details})
           << all_summary_buffer.copy_to(luisa::span{all_summaries})
           << all_detail_buffer.copy_to(luisa::span{all_details})
           << any_summary_buffer.copy_to(luisa::span{any_summaries})
           << any_detail_buffer.copy_to(luisa::span{any_details})
           << synchronize();

    for (auto i = 0u; i < probes.size(); i++) {
        auto &exp = expected[i];
        check_trace_result(exp.name, exp,
                           direct_summaries[i], direct_details[i]);
        check_query_result("ALL query", exp.name, exp,
                           all_summaries[i], all_details[i]);
        check_query_result("ANY query", exp.name, exp,
                           any_summaries[i], any_details[i]);
    }
}

void test_hip_motion_mesh_preprocessing_updates(Device &device) {
    if (device.backend_name() != "hip") { return; }

    auto stream = device.create_stream();

    // The prefix and suffix are deliberately outside the mesh views. The
    // triangle prefix is an invalid index, so accidentally ignoring its
    // four-byte view offset must fail instead of producing a plausible hit.
    auto vertex_storage = device.create_buffer<float4>(11u);
    auto triangle_storage = device.create_buffer<uint>(6u);
    auto vertex_view = vertex_storage.view(1u, 9u);
    auto triangle_view = triangle_storage.view(1u, 3u).as<Triangle>();
    expect(vertex_view.offset_bytes() == sizeof(float4));
    expect(vertex_view.stride() == sizeof(float4));
    expect(triangle_view.offset_bytes() == sizeof(uint));
    expect(triangle_view.offset_bytes() % sizeof(Triangle) != 0u)
        << "triangle view should exercise the public API's four-byte offset";

    AccelOption mesh_option{};
    mesh_option.allow_update = true;
    mesh_option.motion.keyframe_count = 3u;
    mesh_option.motion.time_start = 2.0f;
    mesh_option.motion.time_end = 6.0f;
    auto mesh = device.create_mesh(
        vertex_view, sizeof(float4), triangle_view, mesh_option);
    expect(mesh.vertex_stride() == sizeof(float4));

    AccelOption accel_option{};
    accel_option.allow_update = true;
    auto accel = device.create_accel(accel_option);
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false, 207u);

    Kernel1D trace_motion = [](AccelVar accel, BufferFloat4 probes,
                               BufferUInt4 summaries,
                               BufferFloat4 details) noexcept {
        auto index = dispatch_id().x;
        auto probe = probes.read(index);
        auto ray = make_ray(make_float3(probe.x, probe.y, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);
        auto closest = accel.intersect_motion(ray, probe.z, {});
        auto any = accel.intersect_any_motion(ray, probe.z, {});
        summaries.write(index, make_uint4(
                                   closest->inst, closest->prim,
                                   cast<uint>(any),
                                   cast<uint>(closest->is_triangle())));
        details.write(index, make_float4(
                                 closest->distance(), closest->bary.x,
                                 closest->bary.y,
                                 cast<float>(closest->is_triangle())));
    };
    auto shader = device.compile(trace_motion);
    auto probe_buffer = device.create_buffer<float4>(5u);
    auto summary_buffer = device.create_buffer<uint4>(5u);
    auto detail_buffer = device.create_buffer<float4>(5u);

    const std::array<uint, 6u> triangle_words{
        ~0u, 0u, 1u, 2u, ~0u, ~0u};
    stream << triangle_storage.copy_from(luisa::span{triangle_words});

    auto run_phase = [&](const std::array<float2, 3u> &centers,
                         AccelBuildRequest request,
                         luisa::string_view phase) noexcept {
        auto vertices = make_padded_motion_vertices(centers);
        std::array<float4, 11u> vertex_words{};
        vertex_words.front() = make_float4(
            100.0f, 101.0f, 102.0f, 103.0f);
        for (auto i = 0u; i < vertices.size(); i++) {
            vertex_words[i + 1u] = vertices[i];
        }
        vertex_words.back() = make_float4(
            -100.0f, -101.0f, -102.0f, -103.0f);
        auto probes = make_update_probes(centers);
        std::array<uint4, 5u> summaries{};
        std::array<float4, 5u> details{};
        stream << vertex_storage.copy_from(luisa::span{vertex_words})
               << probe_buffer.copy_from(luisa::span{probes})
               << mesh.build(request)
               << accel.build(request)
               << shader(accel, probe_buffer, summary_buffer, detail_buffer)
                      .dispatch(probes.size())
               << summary_buffer.copy_to(luisa::span{summaries})
               << detail_buffer.copy_to(luisa::span{details})
               << synchronize();
        for (auto i = 0u; i < probes.size(); i++) {
            auto label = luisa::format("{} probe {}", phase, i);
            ExpectedProbe expected{label, true, 0u};
            check_trace_result(label, expected, summaries[i], details[i]);
        }
    };

    // All three phases occupy disjoint y ranges. A stale packed snapshot or
    // stale BLAS update therefore misses rather than accidentally passing.
    run_phase({make_float2(-6.0f, -4.0f),
               make_float2(-4.0f, -4.0f),
               make_float2(-2.0f, -4.0f)},
              AccelBuildRequest::FORCE_BUILD, "initial build");
    run_phase({make_float2(0.0f, 0.0f),
               make_float2(2.0f, 0.0f),
               make_float2(4.0f, 0.0f)},
              AccelBuildRequest::PREFER_UPDATE, "first update");
    run_phase({make_float2(-1.0f, 4.0f),
               make_float2(1.0f, 4.0f),
               make_float2(3.0f, 4.0f)},
              AccelBuildRequest::PREFER_UPDATE, "second update");
}

enum class InvalidMotionMeshInput {
    UNREFERENCED_NON_FINITE_VERTEX,
    INVALID_TRIANGLE_INDEX
};

void run_invalid_motion_mesh_input(
    Device &device, InvalidMotionMeshInput input) {
    auto nan = std::numeric_limits<float>::quiet_NaN();
    std::array<float4, 8u> vertices{
        make_float4(-0.5f, -0.5f, 0.0f, 11.0f),
        make_float4(0.5f, -0.5f, 0.0f, 12.0f),
        make_float4(0.0f, 0.5f, 0.0f, 13.0f),
        make_float4(10.0f, 10.0f, 10.0f, 14.0f),
        make_float4(-0.5f, -0.5f, 0.0f, 21.0f),
        make_float4(0.5f, -0.5f, 0.0f, 22.0f),
        make_float4(0.0f, 0.5f, 0.0f, 23.0f),
        make_float4(10.0f, 10.0f, 10.0f, 24.0f)};
    if (input == InvalidMotionMeshInput::UNREFERENCED_NON_FINITE_VERTEX) {
        // Vertex three is not referenced by the triangle. Validation must
        // still cover every packed position, not just bound inputs.
        vertices[7u].y = nan;
    }
    std::array triangles{
        input == InvalidMotionMeshInput::INVALID_TRIANGLE_INDEX ?
            Triangle{0u, 1u, 4u} :
            Triangle{0u, 1u, 2u}};

    auto stream = device.create_stream();
    auto vertex_buffer = device.create_buffer<float4>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    AccelOption option{};
    option.motion.keyframe_count = 2u;
    option.motion.time_start = 0.0f;
    option.motion.time_end = 1.0f;
    auto mesh = device.create_mesh(
        vertex_buffer, sizeof(float4), triangle_buffer, option);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build(AccelBuildRequest::FORCE_BUILD)
           << synchronize();
}

#if LUISA_TEST_HIP_MOTION_MESH_HAS_EXPECTED_FAILURE_SUBPROCESS
struct ExpectedFailureSubprocessResult {
    bool aborted{};
    int status{};
    luisa::string output;
};

[[nodiscard]] ExpectedFailureSubprocessResult
run_expected_failure_subprocess(const char *executable,
                                const char *backend,
                                const char *mode) {
    auto executable_path = std::filesystem::absolute(executable).string();
    int output_pipe[2]{};
    if (pipe(output_pipe) != 0) { return {false, -errno, {}}; }
    auto pid = fork();
    if (pid < 0) {
        auto error = errno;
        close(output_pipe[0]);
        close(output_pipe[1]);
        return {false, -error, {}};
    }
    if (pid == 0) {
        close(output_pipe[0]);
        if (dup2(output_pipe[1], STDOUT_FILENO) < 0 ||
            dup2(output_pipe[1], STDERR_FILENO) < 0) {
            _exit(126);
        }
        close(output_pipe[1]);
        execl(executable_path.c_str(), executable_path.c_str(),
              backend, mode, static_cast<char *>(nullptr));
        _exit(127);
    }
    close(output_pipe[1]);
    luisa::string output;
    std::array<char, 4096u> buffer{};
    for (;;) {
        auto bytes_read = read(output_pipe[0], buffer.data(), buffer.size());
        if (bytes_read > 0) {
            output.append(buffer.data(), static_cast<size_t>(bytes_read));
        } else if (bytes_read == 0) {
            break;
        } else if (errno != EINTR) {
            output.append(luisa::format("\n[pipe read failed: {}]", errno));
            break;
        }
    }
    close(output_pipe[0]);
    auto status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) { return {false, -errno, std::move(output)}; }
    }
    return {WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
            status, std::move(output)};
}
#endif

}// namespace

int main(int argc, char *argv[]) {
    if (argc > 2 && argv[2] != nullptr) {
        auto mode = luisa::string_view{argv[2]};
        if (mode == invalid_non_finite_mode ||
            mode == invalid_triangle_index_mode) {
            auto dc = luisa::test::create_device_from_ut(argc, argv);
            if (!dc || dc->device.backend_name() != "hip") { return 2; }
            auto input = mode == invalid_non_finite_mode ?
                             InvalidMotionMeshInput::UNREFERENCED_NON_FINITE_VERTEX :
                             InvalidMotionMeshInput::INVALID_TRIANGLE_INDEX;
            run_invalid_motion_mesh_input(dc->device, input);
            // Reaching here means invalid input was silently accepted.
            return 0;
        }
    }

#if LUISA_TEST_HIP_MOTION_MESH_HAS_EXPECTED_FAILURE_SUBPROCESS
    ExpectedFailureSubprocessResult non_finite_failure{};
    ExpectedFailureSubprocessResult invalid_index_failure{};
    auto requested_hip = argc > 1 && argv[1] != nullptr &&
                         luisa::string_view{argv[1]} == "hip";
    if (requested_hip) {
        non_finite_failure = run_expected_failure_subprocess(
            argv[0], argv[1], invalid_non_finite_mode);
        invalid_index_failure = run_expected_failure_subprocess(
            argv[0], argv[1], invalid_triangle_index_mode);
    }
#endif

    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP deforming mesh motion is exact across trace and query paths"_test = [&] {
        test_hip_motion_mesh(dc->device);
    };
    "HIP motion-mesh GPU preprocessing supports views and repeated updates"_test = [&] {
        test_hip_motion_mesh_preprocessing_updates(dc->device);
    };
#if LUISA_TEST_HIP_MOTION_MESH_HAS_EXPECTED_FAILURE_SUBPROCESS
    "HIP motion-mesh GPU preprocessing rejects invalid inputs"_test = [=] {
        if (!requested_hip) { return; }
        expect(non_finite_failure.aborted)
            << luisa::format(
                   "unreferenced non-finite vertex subprocess status was {}; output:\n{}",
                   non_finite_failure.status, non_finite_failure.output);
        expect(non_finite_failure.output.find(
                   "non-finite position component at axis") !=
               luisa::string::npos)
            << luisa::format(
                   "unreferenced non-finite vertex subprocess did not report the "
                   "expected validator; output:\n{}",
                   non_finite_failure.output);
        expect(invalid_index_failure.aborted)
            << luisa::format(
                   "invalid triangle-index subprocess status was {}; output:\n{}",
                   invalid_index_failure.status, invalid_index_failure.output);
        expect(invalid_index_failure.output.find(
                   "invalid local vertex index at corner") !=
               luisa::string::npos)
            << luisa::format(
                   "invalid triangle-index subprocess did not report the expected "
                   "validator; output:\n{}",
                   invalid_index_failure.output);
    };
#endif
}
