// Test for HIP XIR RayQueryPipelineInst code generation.
// This test covers:
// - lowering a DSL ray-query loop to the XIR pipeline instruction
// - surface and procedural candidate handler dispatch
// - query-object state propagation into outlined handlers
// - multiple mutable reference captures shared by both handlers
// - committed world-ray distance after surface and procedural commits

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

void expect_result(uint index, const uint4 &actual0, const uint4 &actual1,
                   const uint4 &expected0, const uint4 &expected1) {
    auto check = [index](luisa::string_view field, uint actual,
                         uint expected) noexcept {
        expect(actual == expected)
            << luisa::format("ray {} {}: got {}, expected {}",
                             index, field, actual, expected);
    };
    check("hit_type", actual0.x, expected0.x);
    check("committed_inst", actual0.y, expected0.y);
    check("committed_prim", actual0.z, expected0.z);
    check("callback_mask", actual0.w, expected0.w);
    check("score", actual1.x, expected1.x);
    check("surface_inst", actual1.y, expected1.y);
    check("procedural_inst", actual1.z, expected1.z);
    check("callback_count", actual1.w, expected1.w);
}

void test_hip_ray_query_pipeline(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP-specific ray-query pipeline test on backend '{}'.",
            device.backend_name());
        return;
    }

    auto stream = device.create_stream();

    // Primitive zero in each BLAS is isolated at x=-2 / x=+2. Primitive one
    // in each BLAS overlaps at x=0, with the triangle strictly nearer than the
    // procedural commit distance. Its surface handler rejects the candidate,
    // so both handlers must observe and mutate the same captured state before
    // the procedural hit is committed.
    const std::array vertices{
        make_float3(-2.5f, -0.5f, 0.0f),
        make_float3(-1.5f, -0.5f, 0.0f),
        make_float3(-2.0f, 0.5f, 0.0f),
        make_float3(-0.5f, -0.5f, 0.5f),
        make_float3(0.5f, -0.5f, 0.5f),
        make_float3(0.0f, 0.5f, 0.5f)};
    const std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u}};
    const std::array aabbs{
        AABB{.packed_min = {1.5f, -0.5f, -2.1f},
             .packed_max = {2.5f, 0.5f, -1.9f}},
        AABB{.packed_min = {-0.5f, -0.5f, -2.1f},
             .packed_max = {0.5f, 0.5f, -1.9f}}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto aabb_buffer = device.create_buffer<AABB>(aabbs.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto procedural = device.create_procedural_primitive(aabb_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    accel.emplace_back(procedural);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << aabb_buffer.copy_from(luisa::span{aabbs})
           << mesh.build()
           << procedural.build()
           << accel.build()
           << synchronize();

    auto result0 = device.create_buffer<uint4>(3u);
    auto result1 = device.create_buffer<uint4>(3u);
    auto result2 = device.create_buffer<float>(3u);

    Kernel1D trace = [](AccelVar accel, BufferUInt4 result0,
                        BufferUInt4 result1,
                        BufferFloat result2,
                        Float procedural_distance) noexcept {
        auto index = dispatch_id().x;
        auto origin_x = ite(index == 0u, -2.0f,
                            ite(index == 1u, 2.0f, 0.0f));
        auto origin = make_float3(origin_x, 0.0f, 1.0f);
        auto ray = make_ray(origin, make_float3(0.0f, 0.0f, -1.0f));

        // All five variables are local allocas captured by reference by both
        // outlined callbacks. Reads after trace() prove mutations survive the
        // RayQueryPipelineInst call rather than merely reaching the callback.
        UInt callback_mask = 0u;
        UInt callback_count = 0u;
        UInt score = 5u + index;
        UInt surface_inst = ~0u;
        UInt procedural_inst = ~0u;
        Float committed_ray_t_max = -1.0f;

        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     callback_mask = callback_mask | 1u;
                                     callback_count += 1u;
                                     score += 10u + hit.prim;
                                     surface_inst = hit.inst;
                                     $if (index != 2u) {
                                         candidate.commit();
                                         committed_ray_t_max = candidate.ray()->t_max();
                                     };
                                 })
                             .on_procedural_candidate(
                                 [&](ProceduralCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     callback_mask = callback_mask | 2u;
                                     callback_count += 1u;
                                     score += 20u + hit.prim;
                                     procedural_inst = hit.inst;
                                     candidate.commit(procedural_distance);
                                     committed_ray_t_max = candidate.ray()->t_max();
                                 })
                             .trace();

        result0.write(index, make_uint4(
                                 committed->hit_type,
                                 committed->inst,
                                 committed->prim,
                                 callback_mask));
        result1.write(index, make_uint4(
                                 score,
                                 surface_inst,
                                 procedural_inst,
                                 callback_count));
        result2.write(index, committed_ray_t_max);
    };

    // This shader naturally needs more than 128 VGPRs on gfx12. Capping it
    // exercises HIP's ShaderOption::max_registers propagation through the
    // linked ray-query call graph as well as correctness under the resulting
    // register allocation/spilling decisions.
    auto shader = device.compile(trace, ShaderOption{.max_registers = 128u});
    std::array<uint4, 3u> host_result0{};
    std::array<uint4, 3u> host_result1{};
    std::array<float, 3u> host_result2{};
    stream << shader(accel, result0, result1, result2, 3.0f).dispatch(3u)
           << result0.copy_to(luisa::span{host_result0})
           << result1.copy_to(luisa::span{host_result1})
           << result2.copy_to(luisa::span{host_result2})
           << synchronize();

    constexpr auto surface = static_cast<uint>(HitType::Surface);
    constexpr auto procedural_hit = static_cast<uint>(HitType::Procedural);
    expect_result(0u, host_result0[0], host_result1[0],
                  make_uint4(surface, 0u, 0u, 1u),
                  make_uint4(15u, 0u, ~0u, 1u));
    expect_result(1u, host_result0[1], host_result1[1],
                  make_uint4(procedural_hit, 1u, 0u, 2u),
                  make_uint4(26u, ~0u, 1u, 1u));
    expect_result(2u, host_result0[2], host_result1[2],
                  make_uint4(procedural_hit, 1u, 1u, 3u),
                  make_uint4(39u, 0u, 1u, 2u));
    expect(std::abs(host_result2[0] - 1.0f) < 1.0e-5f);
    expect(std::abs(host_result2[1] - 3.0f) < 1.0e-5f);
    expect(std::abs(host_result2[2] - 3.0f) < 1.0e-5f);
}

void test_hip_ray_query_paired_triangle_resume(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP-specific paired-triangle ray-query test on backend '{}'.",
            device.backend_name());
        return;
    }
    auto amdgpu_arch = device.query("amdgpu_arch");
    if (amdgpu_arch != "gfx1200" && amdgpu_arch != "gfx1201") {
        LUISA_INFO(
            "Skipping gfx12 compact ray-query state test on AMDGPU architecture '{}'.",
            amdgpu_arch);
        return;
    }

    auto stream = device.create_stream();

    // The first two primitives share an edge and are deliberately ordered so
    // that triangle slot zero is farther away than slot one. HIPRT packs them
    // into one triangle pair (a third primitive is required to enable pairing),
    // which exercises the buffered-hit and leaf-resume state in the gfx12 path.
    const std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(-1.0f, 1.0f, 0.0f),
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(1.0f, 0.0f, 1.0f),
        make_float3(-1.0f, -1.0f, -0.5f),
        make_float3(-1.0f, 1.0f, -0.5f),
        make_float3(1.0f, 0.0f, -0.5f)};
    const std::array triangles{
        Triangle{0u, 1u, 2u}, // far:  t = 1.0
        Triangle{1u, 0u, 3u}, // near: t = 0.5
        Triangle{4u, 5u, 6u}};// later standalone hit: t = 1.5

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    accel.emplace_back(mesh, translation(4.0f, 0.0f, 0.0f), 0xffu, true);
    accel.emplace_back(
        mesh,
        translation(8.0f, 0.0f, 0.0f) * scaling(1.0f, 1.0f, 2.0f),
        0x5au, false);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    constexpr auto case_count = 8u;
    auto metadata = device.create_buffer<uint4>(case_count);
    auto callback_order = device.create_buffer<uint2>(case_count);
    auto committed_detail = device.create_buffer<float4>(case_count);
    auto callback_detail = device.create_buffer<float4>(case_count);
    auto callback_ray_origin_tmin = device.create_buffer<float4>(case_count);
    auto callback_ray_direction_tmax = device.create_buffer<float4>(case_count);

    Kernel1D trace_all = [](AccelVar accel,
                            BufferUInt4 metadata,
                            BufferUInt2 callback_order,
                            BufferFloat4 committed_detail,
                            BufferFloat4 callback_detail,
                            BufferFloat4 callback_ray_origin_tmin,
                            BufferFloat4 callback_ray_direction_tmax) noexcept {
        auto index = dispatch_id().x;
        auto transformed_case = index == 4u;
        auto origin_x = ite(index == 3u, 4.0f,
                            ite(transformed_case, 8.0f, 0.0f));
        auto origin_z = ite(transformed_case, 2.0f, 1.0f);
        auto ray = make_ray(make_float3(origin_x, 0.0f, origin_z),
                            make_float3(0.0f, 0.0f, -1.0f),
                            ite(transformed_case, 0.25f, 0.0f),
                            ite(transformed_case, 4.0f, 2.0f));
        AccelTraceOptions options{
            .visibility_mask = ite(transformed_case, 0x5au, 0xffu)};

        UInt callback_count = 0u;
        UInt first_prim = ~0u;
        UInt second_prim = ~0u;
        Float first_t = -1.0f;
        Float second_t = -1.0f;
        Float first_tmax = -1.0f;
        Float second_tmax = -1.0f;
        Float3 callback_ray_origin = make_float3(-1.0f);
        Float3 callback_ray_direction = make_float3(-1.0f);
        Float callback_ray_tmin = -1.0f;
        Float callback_ray_tmax = -1.0f;
        Float final_callback_tmax = -1.0f;

        auto committed = accel.traverse(ray, options)
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     auto callback_index = callback_count;
                                     auto callback_ray = candidate.ray();
                                     callback_ray_origin = callback_ray->origin();
                                     callback_ray_direction = callback_ray->direction();
                                     callback_ray_tmin = callback_ray->t_min();
                                     callback_ray_tmax = callback_ray->t_max();
                                     $if (callback_index == 0u) {
                                         first_prim = hit->prim;
                                         first_t = hit->distance();
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_prim = hit->prim;
                                             second_t = hit->distance();
                                         };
                                     };
                                     callback_count += 1u;

                                     // Case zero rejects the near triangle and
                                     // commits the buffered far triangle. Case
                                     // one commits the near hit and must suppress
                                     // the buffered far hit. Case two rejects the
                                     // pair and commits a later standalone leaf,
                                     // proving traversal resumes after pending.
                                     // The opaque case must bypass this callback.
                                     $if (index == 0u) {
                                         $if (hit->prim == 0u) {
                                             candidate.commit();
                                         };
                                     }
                                     $else {
                                         $if ((index == 2u) | transformed_case) {
                                             $if (hit->prim == 2u) {
                                                 candidate.commit();
                                             };
                                         }
                                         $else {
                                             candidate.commit();
                                         };
                                     };

                                     auto tmax = candidate.ray()->t_max();
                                     final_callback_tmax = tmax;
                                     $if (callback_index == 0u) {
                                         first_tmax = tmax;
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_tmax = tmax;
                                         };
                                     };
                                 })
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();

        metadata.write(index, make_uint4(
                                  committed->hit_type, committed->inst,
                                  committed->prim, callback_count));
        callback_order.write(index, make_uint2(first_prim, second_prim));
        committed_detail.write(index, make_float4(
                                          committed->distance(), committed->bary.x,
                                          committed->bary.y, final_callback_tmax));
        callback_detail.write(index, make_float4(
                                         first_t, second_t,
                                         first_tmax, second_tmax));
        callback_ray_origin_tmin.write(
            index, make_float4(callback_ray_origin, callback_ray_tmin));
        callback_ray_direction_tmax.write(
            index, make_float4(callback_ray_direction, callback_ray_tmax));
    };

    Kernel1D trace_any = [](AccelVar accel,
                            BufferUInt4 metadata,
                            BufferUInt2 callback_order,
                            BufferFloat4 committed_detail,
                            BufferFloat4 callback_detail) noexcept {
        auto index = dispatch_id().x;
        auto output_index = index + 5u;
        auto origin_x = ite(index == 2u, 4.0f, 0.0f);
        auto ray = make_ray(make_float3(origin_x, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);

        UInt callback_count = 0u;
        UInt first_prim = ~0u;
        UInt second_prim = ~0u;
        Float first_t = -1.0f;
        Float second_t = -1.0f;
        Float first_tmax = -1.0f;
        Float second_tmax = -1.0f;

        auto committed = accel.traverse_any(ray, {})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     auto callback_index = callback_count;
                                     $if (callback_index == 0u) {
                                         first_prim = hit->prim;
                                         first_t = hit->distance();
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_prim = hit->prim;
                                             second_t = hit->distance();
                                         };
                                     };
                                     callback_count += 1u;

                                     // Reject near/commit far, explicitly
                                     // terminate on near without a commit, or
                                     // diagnose an illegal opaque callback.
                                     $if (index == 0u) {
                                         $if (hit->prim == 0u) {
                                             candidate.commit();
                                         };
                                     }
                                     $else {
                                         $if (index == 1u) {
                                             candidate.terminate();
                                         }
                                         $else {
                                             candidate.commit();
                                         };
                                     };

                                     auto tmax = candidate.ray()->t_max();
                                     $if (callback_index == 0u) {
                                         first_tmax = tmax;
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_tmax = tmax;
                                         };
                                     };
                                 })
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();

        metadata.write(output_index, make_uint4(
                                         committed->hit_type, committed->inst,
                                         committed->prim, callback_count));
        callback_order.write(output_index,
                             make_uint2(first_prim, second_prim));
        committed_detail.write(output_index, make_float4(
                                                 committed->distance(), committed->bary.x,
                                                 committed->bary.y, 0.0f));
        callback_detail.write(output_index, make_float4(
                                                first_t, second_t,
                                                first_tmax, second_tmax));
    };

    auto all_shader = device.compile(trace_all);
    auto any_shader = device.compile(trace_any);
    std::array<uint4, case_count> host_metadata{};
    std::array<uint2, case_count> host_callback_order{};
    std::array<float4, case_count> host_committed_detail{};
    std::array<float4, case_count> host_callback_detail{};
    std::array<float4, case_count> host_callback_ray_origin_tmin{};
    std::array<float4, case_count> host_callback_ray_direction_tmax{};
    stream << all_shader(accel, metadata, callback_order,
                         committed_detail, callback_detail,
                         callback_ray_origin_tmin, callback_ray_direction_tmax)
                  .dispatch(5u)
           << any_shader(accel, metadata, callback_order,
                         committed_detail, callback_detail)
                  .dispatch(3u)
           << metadata.copy_to(luisa::span{host_metadata})
           << callback_order.copy_to(luisa::span{host_callback_order})
           << committed_detail.copy_to(luisa::span{host_committed_detail})
           << callback_detail.copy_to(luisa::span{host_callback_detail})
           << callback_ray_origin_tmin.copy_to(
                  luisa::span{host_callback_ray_origin_tmin})
           << callback_ray_direction_tmax.copy_to(
                  luisa::span{host_callback_ray_direction_tmax})
           << synchronize();

    constexpr auto surface = static_cast<uint>(HitType::Surface);
    constexpr auto miss = static_cast<uint>(HitType::Miss);
    const std::array expected_metadata{
        make_uint4(surface, 0u, 0u, 2u),
        make_uint4(surface, 0u, 1u, 1u),
        make_uint4(surface, 0u, 2u, 3u),
        make_uint4(surface, 1u, 1u, 0u),
        make_uint4(surface, 2u, 2u, 3u),
        make_uint4(surface, 0u, 0u, 2u),
        make_uint4(miss, ~0u, ~0u, 1u),
        make_uint4(surface, 1u, 1u, 0u)};
    const std::array expected_callback_order{
        make_uint2(1u, 0u),
        make_uint2(1u, ~0u),
        make_uint2(1u, 0u),
        make_uint2(~0u, ~0u),
        make_uint2(1u, 0u),
        make_uint2(1u, 0u),
        make_uint2(1u, ~0u),
        make_uint2(~0u, ~0u)};
    const std::array expected_committed_detail{
        make_float4(1.0f, 0.25f, 0.5f, 0.0f),
        make_float4(0.5f, 0.25f, 0.5f, 0.0f),
        make_float4(1.5f, 0.25f, 0.5f, 0.0f),
        make_float4(0.5f, 0.25f, 0.5f, 0.0f),
        make_float4(3.0f, 0.25f, 0.5f, 3.0f),
        make_float4(1.0f, 0.25f, 0.5f, 0.0f),
        make_float4(2.0f, 0.0f, 0.0f, 0.0f),
        make_float4(0.5f, 0.25f, 0.5f, 0.0f)};
    const std::array expected_callback_detail{
        make_float4(0.5f, 1.0f, 2.0f, 1.0f),
        make_float4(0.5f, -1.0f, 0.5f, -1.0f),
        make_float4(0.5f, 1.0f, 2.0f, 2.0f),
        make_float4(-1.0f),
        make_float4(1.0f, 2.0f, 4.0f, 4.0f),
        make_float4(0.5f, 1.0f, 2.0f, 1.0f),
        make_float4(0.5f, -1.0f, 2.0f, -1.0f),
        make_float4(-1.0f)};

    auto check_uint = [](uint case_index, luisa::string_view field,
                         uint actual, uint expected) noexcept {
        expect(actual == expected)
            << luisa::format("paired-triangle case {} {}: got {}, expected {}",
                             case_index, field, actual, expected);
    };
    auto check_float = [](uint case_index, luisa::string_view field,
                          float actual, float expected) noexcept {
        expect(std::abs(actual - expected) < 1.0e-5f)
            << luisa::format("paired-triangle case {} {}: got {}, expected {}",
                             case_index, field, actual, expected);
    };
    for (auto i = 0u; i < case_count; i++) {
        auto actual_meta = host_metadata[i];
        auto expected_meta = expected_metadata[i];
        check_uint(i, "hit_type", actual_meta.x, expected_meta.x);
        check_uint(i, "instance", actual_meta.y, expected_meta.y);
        check_uint(i, "primitive", actual_meta.z, expected_meta.z);
        check_uint(i, "callback_count", actual_meta.w, expected_meta.w);

        auto actual_order = host_callback_order[i];
        auto expected_order = expected_callback_order[i];
        check_uint(i, "first_callback_primitive",
                   actual_order.x, expected_order.x);
        check_uint(i, "second_callback_primitive",
                   actual_order.y, expected_order.y);

        // Miss payload distance/barycentrics are not part of the semantic
        // contract; only validate committed detail for actual hits.
        if (actual_meta.x != miss) {
            auto actual_committed = host_committed_detail[i];
            auto expected_committed = expected_committed_detail[i];
            check_float(i, "committed_distance",
                        actual_committed.x, expected_committed.x);
            check_float(i, "committed_bary_u",
                        actual_committed.y, expected_committed.y);
            check_float(i, "committed_bary_v",
                        actual_committed.z, expected_committed.z);
        }

        auto actual_callback = host_callback_detail[i];
        auto expected_callback = expected_callback_detail[i];
        check_float(i, "first_callback_distance",
                    actual_callback.x, expected_callback.x);
        check_float(i, "second_callback_distance",
                    actual_callback.y, expected_callback.y);
        check_float(i, "first_callback_tmax",
                    actual_callback.z, expected_callback.z);
        check_float(i, "second_callback_tmax",
                    actual_callback.w, expected_callback.w);
    }

    // The traversal ray is transformed into BLAS space, but callback accessors
    // must keep exposing the original world-space ray across all three yields.
    constexpr auto transformed_case = 4u;
    const auto transformed_origin_tmin =
        host_callback_ray_origin_tmin[transformed_case];
    const auto transformed_direction_tmax =
        host_callback_ray_direction_tmax[transformed_case];
    check_float(transformed_case, "world_origin_x",
                transformed_origin_tmin.x, 8.0f);
    check_float(transformed_case, "world_origin_y",
                transformed_origin_tmin.y, 0.0f);
    check_float(transformed_case, "world_origin_z",
                transformed_origin_tmin.z, 2.0f);
    check_float(transformed_case, "world_tmin",
                transformed_origin_tmin.w, 0.25f);
    check_float(transformed_case, "world_direction_x",
                transformed_direction_tmax.x, 0.0f);
    check_float(transformed_case, "world_direction_y",
                transformed_direction_tmax.y, 0.0f);
    check_float(transformed_case, "world_direction_z",
                transformed_direction_tmax.z, -1.0f);
    check_float(transformed_case, "world_tmax_before_commit",
                transformed_direction_tmax.w, 4.0f);
    check_float(transformed_case, "world_tmax_after_commit",
                host_committed_detail[transformed_case].w, 3.0f);
}

void test_hip_ray_query_any_automatic_termination(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP-specific ANY ray-query termination test on backend '{}'.",
            device.backend_name());
        return;
    }
    auto amdgpu_arch = device.query("amdgpu_arch");
    if (amdgpu_arch != "gfx1200" && amdgpu_arch != "gfx1201") {
        LUISA_INFO(
            "Skipping gfx12 compact ray-query termination test on AMDGPU architecture '{}'.",
            amdgpu_arch);
        return;
    }

    auto stream = device.create_stream();
    const std::array aabbs{
        AABB{.packed_min = {-1.0f, -1.0f, 0.5f},
             .packed_max = {1.0f, 1.0f, 1.5f}},
        AABB{.packed_min = {-1.0f, -1.0f, 0.5f},
             .packed_max = {1.0f, 1.0f, 1.5f}}};
    auto aabb_buffer = device.create_buffer<AABB>(aabbs.size());
    auto procedural = device.create_procedural_primitive(aabb_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(procedural);
    stream << aabb_buffer.copy_from(luisa::span{aabbs})
           << procedural.build()
           << accel.build()
           << synchronize();

    auto metadata = device.create_buffer<uint4>(2u);
    auto callback_order = device.create_buffer<uint2>(2u);
    auto detail = device.create_buffer<float2>(2u);

    Kernel1D trace_all = [](AccelVar accel, BufferUInt4 metadata,
                            BufferUInt2 callback_order,
                            BufferFloat2 detail) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 2.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 3.0f);
        UInt callback_count = 0u;
        UInt first_prim = ~0u;
        UInt second_prim = ~0u;
        Float final_tmax = -1.0f;
        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate(
                                 [](SurfaceCandidate &) noexcept {})
                             .on_procedural_candidate(
                                 [&](ProceduralCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     auto callback_index = callback_count;
                                     $if (callback_index == 0u) {
                                         first_prim = hit->prim;
                                         candidate.commit(1.5f);
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_prim = hit->prim;
                                         };
                                         candidate.commit(1.0f);
                                     };
                                     callback_count += 1u;
                                     final_tmax = candidate.ray()->t_max();
                                 })
                             .trace();
        metadata.write(0u, make_uint4(
                               committed->hit_type, committed->inst,
                               committed->prim, callback_count));
        callback_order.write(0u, make_uint2(first_prim, second_prim));
        detail.write(0u, make_float2(committed->distance(), final_tmax));
    };

    Kernel1D trace_any = [](AccelVar accel, BufferUInt4 metadata,
                            BufferUInt2 callback_order,
                            BufferFloat2 detail) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 2.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 3.0f);
        UInt callback_count = 0u;
        UInt first_prim = ~0u;
        UInt second_prim = ~0u;
        Float final_tmax = -1.0f;
        auto committed = accel.traverse_any(ray, {})
                             .on_surface_candidate(
                                 [](SurfaceCandidate &) noexcept {})
                             .on_procedural_candidate(
                                 [&](ProceduralCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     auto callback_index = callback_count;
                                     $if (callback_index == 0u) {
                                         first_prim = hit->prim;
                                     }
                                     $else {
                                         $if (callback_index == 1u) {
                                             second_prim = hit->prim;
                                         };
                                     };
                                     callback_count += 1u;
                                     candidate.commit(1.5f);
                                     final_tmax = candidate.ray()->t_max();
                                 })
                             .trace();
        metadata.write(1u, make_uint4(
                               committed->hit_type, committed->inst,
                               committed->prim, callback_count));
        callback_order.write(1u, make_uint2(first_prim, second_prim));
        detail.write(1u, make_float2(committed->distance(), final_tmax));
    };

    auto all_shader = device.compile(trace_all);
    auto any_shader = device.compile(trace_any);
    std::array<uint4, 2u> host_metadata{};
    std::array<uint2, 2u> host_callback_order{};
    std::array<float2, 2u> host_detail{};
    stream << all_shader(accel, metadata, callback_order, detail).dispatch(1u)
           << any_shader(accel, metadata, callback_order, detail).dispatch(1u)
           << metadata.copy_to(luisa::span{host_metadata})
           << callback_order.copy_to(luisa::span{host_callback_order})
           << detail.copy_to(luisa::span{host_detail})
           << synchronize();

    constexpr auto procedural_hit = static_cast<uint>(HitType::Procedural);
    expect(host_metadata[0].x == procedural_hit);
    expect(host_metadata[0].y == 0u);
    expect(host_metadata[0].w == 2u)
        << "ALL query must continue to the second procedural candidate";
    expect(host_callback_order[0].x < 2u);
    expect(host_callback_order[0].y < 2u);
    expect(host_callback_order[0].x != host_callback_order[0].y);
    expect(host_metadata[0].z == host_callback_order[0].y)
        << "ALL query must retain the second, closer procedural commit";
    expect(std::abs(host_detail[0].x - 1.0f) < 1.0e-5f);
    expect(std::abs(host_detail[0].y - 1.0f) < 1.0e-5f);

    expect(host_metadata[1].x == procedural_hit);
    expect(host_metadata[1].y == 0u);
    expect(host_metadata[1].w == 1u)
        << "ANY commit must terminate before the second eligible candidate";
    expect(host_callback_order[1].x < 2u);
    expect(host_callback_order[1].y == ~0u);
    expect(host_metadata[1].z == host_callback_order[1].x);
    expect(std::abs(host_detail[1].x - 1.5f) < 1.0e-5f);
    expect(std::abs(host_detail[1].y - 1.5f) < 1.0e-5f);
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP ray-query pipeline captures and commits"_test = [&] {
        test_hip_ray_query_pipeline(dc->device);
    };
    "HIP ray-query paired-triangle resume state"_test = [&] {
        test_hip_ray_query_paired_triangle_resume(dc->device);
    };
    "HIP ray-query ANY commit terminates automatically"_test = [&] {
        test_hip_ray_query_any_automatic_termination(dc->device);
    };
}
