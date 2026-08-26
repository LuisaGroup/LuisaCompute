// Focused strict-AIR test for Metal ray-query lowering.
// This test covers:
// - QueryAll candidate iteration, rejection, and selective triangle commitment
// - QueryAny triangle commitment followed by explicit termination
// - candidate and committed-hit instance, primitive, barycentric, and distance data
// - visibility-mask filtering and committed misses

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

void test_metal_xir_air_ray_query(Device &device) {
    log_level_verbose();

    // Two identically shaped triangles are placed at z = 0 and z = -1. A ray
    // starting at z = 1 therefore reaches primitive 0 at t = 1 and primitive
    // 1 at t = 2. The instance is non-opaque so both hits enter the callbacks.
    constexpr std::array vertices{
        float3{-1.0f, -1.0f, 0.0f},
        float3{1.0f, -1.0f, 0.0f},
        float3{0.0f, 1.0f, 0.0f},
        float3{-1.0f, -1.0f, -1.0f},
        float3{1.0f, -1.0f, -1.0f},
        float3{0.0f, 1.0f, -1.0f}};
    constexpr std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f),
                       instance_visibility, false);

    auto result_buffer = device.create_buffer<uint>(18u);
    auto hit_data_buffer = device.create_buffer<float4>(4u);

    Kernel1D ray_query_kernel = [](AccelVar accel,
                                   BufferUInt results,
                                   BufferFloat4 hit_data) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 10.0f);

        // QueryAll: reject primitive 0 and commit primitive 1. This proves that
        // traversal continues after a rejected candidate and that the final
        // committed hit is the explicitly accepted farther triangle.
        UInt all_candidate_count = 0u;
        UInt all_candidate_inst = ~0u;
        UInt all_candidate_prim = ~0u;
        Float2 all_candidate_bary = make_float2(-1.0f);
        Float all_candidate_t = -1.0f;
        Var<CommittedHit> all_hit = accel.traverse(
                                             ray, {.visibility_mask = 0x08u})
                                        .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                            auto hit = candidate.hit();
                                            all_candidate_count += 1u;
                                            $if (hit.prim == 1u) {
                                                all_candidate_inst = hit.inst;
                                                all_candidate_prim = hit.prim;
                                                all_candidate_bary = hit.bary;
                                                all_candidate_t = hit->distance();
                                                candidate.commit();
                                            };
                                        })
                                        .trace();

        results.write(0u, all_candidate_count);
        results.write(1u, all_candidate_inst);
        results.write(2u, all_candidate_prim);
        results.write(3u, all_hit.hit_type);
        results.write(4u, all_hit.inst);
        results.write(5u, all_hit.prim);
        hit_data.write(0u, make_float4(all_candidate_bary, all_candidate_t, 0.0f));
        hit_data.write(1u, make_float4(all_hit.bary, all_hit->distance(), 0.0f));

        // QueryAll: observe both candidates but reject both. No committed hit
        // must be synthesized merely because candidates were reported.
        UInt reject_candidate_count = 0u;
        Var<CommittedHit> reject_hit = accel.traverse(
                                                ray, {.visibility_mask = 0x08u})
                                           .on_surface_candidate([&](SurfaceCandidate &) noexcept {
                                               reject_candidate_count += 1u;
                                           })
                                           .trace();
        results.write(6u, reject_candidate_count);
        results.write(7u, ite(reject_hit->miss(), 1u, 0u));
        results.write(8u, reject_hit.hit_type);

        // QueryAny: accept the first reported candidate and explicitly stop.
        // Traversal order need not be assumed; the candidate and committed
        // payloads must match exactly whichever primitive was returned first.
        UInt any_candidate_count = 0u;
        UInt any_candidate_inst = ~0u;
        UInt any_candidate_prim = ~0u;
        Float2 any_candidate_bary = make_float2(-1.0f);
        Float any_candidate_t = -1.0f;
        Var<CommittedHit> any_hit = accel.traverse_any(
                                             ray, {.visibility_mask = 0x08u})
                                        .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                            auto hit = candidate.hit();
                                            any_candidate_count += 1u;
                                            any_candidate_inst = hit.inst;
                                            any_candidate_prim = hit.prim;
                                            any_candidate_bary = hit.bary;
                                            any_candidate_t = hit->distance();
                                            candidate.commit();
                                            candidate.terminate();
                                        })
                                        .trace();
        results.write(9u, any_candidate_count);
        results.write(10u, any_candidate_inst);
        results.write(11u, any_candidate_prim);
        results.write(12u, any_hit.hit_type);
        results.write(13u, any_hit.inst);
        results.write(14u, any_hit.prim);
        hit_data.write(2u, make_float4(any_candidate_bary, any_candidate_t, 0.0f));
        hit_data.write(3u, make_float4(any_hit.bary, any_hit->distance(), 0.0f));

        // A non-overlapping mask must suppress the callback and produce a miss,
        // even though the handler would commit any candidate it observed.
        UInt filtered_candidate_count = 0u;
        Var<CommittedHit> filtered_hit = accel.traverse(
                                                  ray, {.visibility_mask = 0x01u})
                                             .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                                 filtered_candidate_count += 1u;
                                                 candidate.commit();
                                             })
                                             .trace();
        results.write(15u, filtered_candidate_count);
        results.write(16u, ite(filtered_hit->miss(), 1u, 0u));
        results.write(17u, filtered_hit.hit_type);
    };
    auto ray_query_shader = device.compile(ray_query_kernel);

    std::array<uint, 18u> results{};
    std::array<float4, 4u> hit_data{};

    auto stream = device.create_stream();
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << ray_query_shader(accel, result_buffer, hit_data_buffer).dispatch(1u)
           << result_buffer.copy_to(luisa::span{results})
           << hit_data_buffer.copy_to(luisa::span{hit_data})
           << synchronize();

    constexpr auto surface_hit = static_cast<uint>(HitType::Surface);
    constexpr auto miss = static_cast<uint>(HitType::Miss);
    constexpr auto epsilon = 1.0e-5f;

    expect(results[0] == 2u) << "QueryAll should report both triangles";
    expect(results[1] == 0u) << "accepted QueryAll candidate should use instance zero";
    expect(results[2] == 1u) << "QueryAll should commit only primitive one";
    expect(results[3] == surface_hit) << "QueryAll should commit a surface hit";
    expect(results[4] == 0u) << "committed QueryAll hit should use instance zero";
    expect(results[5] == 1u) << "committed QueryAll hit should be primitive one";
    expect(std::abs(hit_data[0].x - 0.25f) < epsilon)
        << "unexpected accepted QueryAll candidate barycentric x";
    expect(std::abs(hit_data[0].y - 0.5f) < epsilon)
        << "unexpected accepted QueryAll candidate barycentric y";
    expect(std::abs(hit_data[0].z - 2.0f) < epsilon)
        << "unexpected accepted QueryAll candidate distance";
    expect(std::abs(hit_data[1].x - hit_data[0].x) < epsilon &&
           std::abs(hit_data[1].y - hit_data[0].y) < epsilon &&
           std::abs(hit_data[1].z - hit_data[0].z) < epsilon)
        << "QueryAll committed payload should match its accepted candidate";

    expect(results[6] == 2u) << "rejecting QueryAll should still visit both triangles";
    expect(results[7] == 1u) << "rejecting every candidate should produce a miss";
    expect(results[8] == miss) << "rejected QueryAll hit type should be Miss";

    expect(results[9] == 1u) << "terminated QueryAny should report one candidate";
    expect(results[10] == 0u) << "QueryAny candidate should use instance zero";
    expect(results[11] <= 1u) << "QueryAny candidate primitive should be valid";
    expect(results[12] == surface_hit) << "QueryAny should commit a surface hit";
    expect(results[13] == results[10])
        << "QueryAny committed instance should match the accepted candidate";
    expect(results[14] == results[11])
        << "QueryAny committed primitive should match the accepted candidate";
    auto any_expected_t = results[11] == 0u ? 1.0f : 2.0f;
    expect(std::abs(hit_data[2].x - 0.25f) < epsilon)
        << "unexpected QueryAny candidate barycentric x";
    expect(std::abs(hit_data[2].y - 0.5f) < epsilon)
        << "unexpected QueryAny candidate barycentric y";
    expect(std::abs(hit_data[2].z - any_expected_t) < epsilon)
        << "unexpected QueryAny candidate distance";
    expect(std::abs(hit_data[3].x - hit_data[2].x) < epsilon &&
           std::abs(hit_data[3].y - hit_data[2].y) < epsilon &&
           std::abs(hit_data[3].z - hit_data[2].z) < epsilon)
        << "QueryAny committed payload should match its accepted candidate";

    expect(results[15] == 0u) << "filtered QueryAll should report no candidates";
    expect(results[16] == 1u) << "filtered QueryAll should produce a miss";
    expect(results[17] == miss) << "filtered QueryAll hit type should be Miss";
}

void test_metal_xir_air_procedural_ray_query(Device &device) {
    std::array<AABB, 1u> bounds{};
    bounds[0].packed_min = {-0.5f, -0.5f, -0.1f};
    bounds[0].packed_max = {0.5f, 0.5f, 0.1f};

    auto aabb_buffer = device.create_buffer<AABB>(bounds.size());
    auto primitive = device.create_procedural_primitive(aabb_buffer.view());
    auto accel = device.create_accel();
    accel.emplace_back(primitive);

    auto result_buffer = device.create_buffer<uint>(9u);
    auto hit_data_buffer = device.create_buffer<float4>(3u);

    Kernel1D kernel = [](AccelVar accel,
                         BufferUInt results,
                         BufferFloat4 hit_data) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 10.0f);
        UInt candidate_count = 0u;
        UInt candidate_inst = ~0u;
        UInt candidate_prim = ~0u;
        Float4 world_origin_and_t_min = make_float4(-1.0f);
        Float4 world_direction_and_t_max = make_float4(-1.0f);
        Var<CommittedHit> hit = accel.traverse(ray, {})
                                    .on_procedural_candidate(
                                        [&](ProceduralCandidate &candidate) noexcept {
                                            auto candidate_hit = candidate.hit();
                                            auto world_ray = candidate.ray();
                                            candidate_count += 1u;
                                            candidate_inst = candidate_hit.inst;
                                            candidate_prim = candidate_hit.prim;
                                            world_origin_and_t_min = make_float4(
                                                world_ray->origin(), world_ray->t_min());
                                            world_direction_and_t_max = make_float4(
                                                world_ray->direction(), world_ray->t_max());
                                            candidate.commit(0.9f);
                                        })
                                    .trace();
        results.write(0u, candidate_count);
        results.write(1u, candidate_inst);
        results.write(2u, candidate_prim);
        results.write(3u, hit.hit_type);
        results.write(4u, hit.inst);
        results.write(5u, hit.prim);
        results.write(6u, ite(hit->is_procedural(), 1u, 0u));
        hit_data.write(0u, world_origin_and_t_min);
        hit_data.write(1u, world_direction_and_t_max);
        hit_data.write(2u, make_float4(hit.bary, hit->distance(), 0.0f));

        UInt invalid_candidate_count = 0u;
        Var<CommittedHit> invalid_hit = accel.traverse(ray, {})
                                            .on_procedural_candidate(
                                                [&](ProceduralCandidate &candidate) noexcept {
                                                    invalid_candidate_count += 1u;
                                                    candidate.commit(-1.0f);
                                                })
                                            .trace();
        results.write(7u, invalid_candidate_count);
        results.write(8u, ite(invalid_hit->miss(), 1u, 0u));
    };
    auto shader = device.compile(kernel);

    std::array<uint, 9u> results{};
    std::array<float4, 3u> hit_data{};
    auto stream = device.create_stream();
    stream << aabb_buffer.copy_from(luisa::span{bounds})
           << primitive.build()
           << accel.build()
           << shader(accel, result_buffer, hit_data_buffer).dispatch(1u)
           << result_buffer.copy_to(luisa::span{results})
           << hit_data_buffer.copy_to(luisa::span{hit_data})
           << synchronize();

    constexpr auto procedural = static_cast<uint>(HitType::Procedural);
    constexpr auto epsilon = 1.0e-5f;
    expect(results[0] == 1u) << "procedural QueryAll should report one AABB";
    expect(results[1] == 0u && results[2] == 0u)
        << "unexpected procedural candidate IDs";
    expect(results[3] == procedural && results[6] == 1u)
        << "procedural QueryAll should commit a procedural hit";
    expect(results[4] == 0u && results[5] == 0u)
        << "unexpected committed procedural IDs";
    expect(std::abs(hit_data[0].x) < epsilon &&
           std::abs(hit_data[0].y) < epsilon &&
           std::abs(hit_data[0].z - 1.0f) < epsilon &&
           std::abs(hit_data[0].w) < epsilon)
        << "unexpected procedural world-ray origin or minimum distance";
    expect(std::abs(hit_data[1].x) < epsilon &&
           std::abs(hit_data[1].y) < epsilon &&
           std::abs(hit_data[1].z + 1.0f) < epsilon &&
           std::abs(hit_data[1].w - 10.0f) < epsilon)
        << "unexpected procedural world-ray direction or maximum distance";
    expect(std::abs(hit_data[2].z - 0.9f) < epsilon)
        << "unexpected committed procedural distance";
    expect(results[7] == 1u)
        << "invalid procedural commit should still observe the candidate";
    expect(results[8] == 1u)
        << "out-of-range procedural distance must not commit a hit";
}

void test_metal_xir_air_curve_ray_query(Device &device) {
    constexpr auto basis = CurveBasis::PIECEWISE_LINEAR;
    constexpr std::array control_points{
        float4{-1.0f, 0.0f, 0.0f, 0.2f},
        float4{1.0f, 0.0f, 0.0f, 0.2f}};
    constexpr std::array segments{0u};

    auto control_point_buffer =
        device.create_buffer<float4>(control_points.size());
    auto segment_buffer = device.create_buffer<uint>(segments.size());
    auto curve = device.create_curve(
        basis, control_point_buffer, segment_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(
        curve, make_float4x4(1.0f), 0xffu, false);
    auto result_buffer = device.create_buffer<uint>(7u);
    auto hit_data_buffer = device.create_buffer<float4>(2u);

    Kernel1D kernel = [](AccelVar accel, BufferUInt results,
                         BufferFloat4 hit_data) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
        UInt candidate_count = 0u;
        UInt candidate_is_curve = 0u;
        Var<TriangleHit> candidate_hit{};
        Var<CommittedHit> committed =
            accel.traverse(
                     ray,
                     {.curve_bases = {basis}})
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        candidate_count += 1u;
                        candidate_hit = candidate.hit();
                        candidate_is_curve = ite(
                            candidate_hit->is_curve(), 1u, 0u);
                        candidate.commit();
                    })
                .trace();
        results.write(0u, candidate_count);
        results.write(1u, candidate_is_curve);
        results.write(2u, candidate_hit.inst);
        results.write(3u, candidate_hit.prim);
        results.write(4u, committed.hit_type);
        results.write(5u, committed.inst);
        results.write(6u, committed.prim);
        hit_data.write(
            0u, make_float4(
                    candidate_hit.bary,
                    candidate_hit->distance(), 0.0f));
        hit_data.write(
            1u, make_float4(
                    committed.bary,
                    committed->distance(), 0.0f));
    };
    auto shader = device.compile(kernel);

    std::array<uint, 7u> results{};
    std::array<float4, 2u> hit_data{};
    auto stream = device.create_stream();
    stream << control_point_buffer.copy_from(
                  luisa::span{control_points})
           << segment_buffer.copy_from(luisa::span{segments})
           << curve.build()
           << accel.build()
           << shader(accel, result_buffer, hit_data_buffer).dispatch(1u)
           << result_buffer.copy_to(luisa::span{results})
           << hit_data_buffer.copy_to(luisa::span{hit_data})
           << synchronize();

    constexpr auto surface = static_cast<uint>(HitType::Surface);
    constexpr auto epsilon = 1.0e-4f;
    expect(results[0] == 1u)
        << "curve QueryAll should report one curve candidate";
    expect(results[1] == 1u)
        << "curve candidate must use Luisa surface/curve semantics";
    expect(results[2] == 0u && results[3] == 0u)
        << "unexpected curve candidate IDs";
    expect(results[4] == surface && results[5] == 0u &&
           results[6] == 0u)
        << "curve commit must produce a surface committed hit";
    expect(std::abs(hit_data[0].x - 0.5f) < epsilon &&
           std::abs(hit_data[0].y + 1.0f) < epsilon &&
           std::abs(hit_data[0].z - 0.8f) < epsilon)
        << "unexpected curve candidate parameter or distance";
    expect(std::abs(hit_data[1].x - hit_data[0].x) < epsilon &&
           std::abs(hit_data[1].y - hit_data[0].y) < epsilon &&
           std::abs(hit_data[1].z - hit_data[0].z) < epsilon)
        << "committed curve payload should match the candidate";
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_metal_xir_air_ray_query(dc->device);
    test_metal_xir_air_procedural_ray_query(dc->device);
    test_metal_xir_air_curve_ray_query(dc->device);
}
