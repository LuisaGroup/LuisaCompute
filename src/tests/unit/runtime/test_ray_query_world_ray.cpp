// Ray-query world-ray state regression.
//
// The ray's world-space origin, direction, and TMin are immutable. TMax is the
// current committed traversal bound: it starts at the initialization value and
// shrinks to the accepted hit distance. Vulkan must preserve the initialization
// value explicitly because SPIR-V committed IntersectionT is undefined while
// the committed intersection type is None.

#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto tolerance = 1.0e-5f;

void expect_near(float actual, float expected,
                 luisa::string_view label) noexcept {
    expect(std::abs(actual - expected) < tolerance)
        << luisa::format("{}: got {}, expected {}", label, actual, expected);
}

void test_ray_query_world_ray(Device &device) {
    auto stream = device.create_stream();

    const std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    const auto transform = make_float4x4(
        make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 1.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 2.0f, 0.0f),
        make_float4(0.0f, 0.0f, 1.0f, 1.0f));
    // Non-opaque is intentional: the callback must observe the no-commit state.
    // The non-identity instance is intentional: Embree invokes geometry
    // callbacks with an object-space ray, while Luisa's candidate.ray()
    // contract exposes the immutable world-space ray on every backend.
    accel.emplace_back(mesh, transform, 0xffu, false);

    auto result_buffer = device.create_buffer<float4>(5u);
    Kernel1D trace = [](AccelVar accel, BufferFloat4 results) noexcept {
        constexpr auto initial_origin = make_float3(0.125f, -0.125f, 3.0f);
        constexpr auto initial_direction = make_float3(0.0f, 0.0f, -2.0f);
        constexpr auto initial_t_min = 0.25f;
        constexpr auto initial_t_max = 7.5f;
        auto ray = make_ray(initial_origin, initial_direction,
                            initial_t_min, initial_t_max);
        Float candidate_t = -1.0f;
        UInt callback_count = 0u;
        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto before = candidate.ray();
                                     results.write(0u, make_float4(
                                                           before->origin(), before->t_min()));
                                     results.write(1u, make_float4(
                                                           before->direction(), before->t_max()));
                                     candidate_t = candidate.hit()->committed_ray_t;
                                     callback_count += 1u;
                                     candidate.commit();
                                     auto after = candidate.ray();
                                     results.write(2u, make_float4(
                                                           after->origin(), after->t_min()));
                                     results.write(3u, make_float4(
                                                           after->direction(), after->t_max()));
                                 })
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();
        results.write(4u, make_float4(
                              candidate_t,
                              cast<float>(committed->hit_type),
                              cast<float>(callback_count), 0.0f));
    };

    auto shader = device.compile(trace);
    std::array<float4, 5u> host_results{};
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << result_buffer.copy_from(luisa::span{host_results})
           << shader(accel, result_buffer).dispatch(1u)
           << result_buffer.copy_to(luisa::span{host_results})
           << synchronize();

    auto check_immutable_ray_fields = [&](const float4 &origin_tmin,
                                          const float4 &direction_tmax,
                                          luisa::string_view phase) noexcept {
        expect_near(origin_tmin.x, 0.125f, luisa::format("{} origin.x", phase));
        expect_near(origin_tmin.y, -0.125f, luisa::format("{} origin.y", phase));
        expect_near(origin_tmin.z, 3.0f, luisa::format("{} origin.z", phase));
        expect_near(origin_tmin.w, 0.25f, luisa::format("{} t_min", phase));
        expect_near(direction_tmax.x, 0.0f, luisa::format("{} direction.x", phase));
        expect_near(direction_tmax.y, 0.0f, luisa::format("{} direction.y", phase));
        expect_near(direction_tmax.z, -2.0f, luisa::format("{} direction.z", phase));
    };
    check_immutable_ray_fields(host_results[0], host_results[1], "before commit");
    check_immutable_ray_fields(host_results[2], host_results[3], "after commit");

    expect_near(host_results[1].w, 7.5f, "pre-commit t_max");
    expect_near(host_results[3].w, host_results[4].x, "post-commit t_max");
    expect_near(host_results[4].x, 1.0f, "candidate distance");
    expect(static_cast<uint>(host_results[4].y) ==
           static_cast<uint>(HitType::Surface));
    expect(static_cast<uint>(host_results[4].z) == 1u);
}

void test_ray_query_commits_closest_surface(Device &device) {
    auto stream = device.create_stream();

    // A ray exactly on a shared sloped edge exercises the watertight ownership
    // rule. The values are reduced from a smooth sphere cap where the fallback
    // backend previously skipped both front triangles and returned the rear
    // cap. Candidate visitation/edge ownership is backend-defined, but the
    // closest committed distance is not.
    const std::array vertices{
        make_float3(0.026116198f, 0.031822680f, 0.41797757f),
        make_float3(0.0f, 0.0f, 0.42f),
        make_float3(0.029109610f, 0.029109610f, 0.41797757f),
        make_float3(0.031822680f, 0.026116198f, 0.41797757f),
        make_float3(0.026116198f, 0.031822680f, -0.41797757f),
        make_float3(0.0f, 0.0f, -0.42f),
        make_float3(0.029109610f, 0.029109610f, -0.41797757f),
        make_float3(0.031822680f, 0.026116198f, -0.41797757f)};
    const std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{2u, 1u, 3u},
        Triangle{5u, 4u, 6u},
        Triangle{5u, 6u, 7u}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);

    auto result_buffer = device.create_buffer<float4>(1u);
    Kernel1D trace = [](AccelVar accel, BufferFloat4 results) noexcept {
        auto ray = make_ray(make_float3(0.017204285f, 0.017204285f, 2.9f),
                            make_float3(0.0f, 0.0f, -1.0f));
        UInt source_instance = ~0u;
        UInt source_primitive = ~0u;
        UInt callback_count = 0u;
        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     const auto hit = candidate.hit();
                                     callback_count += 1u;
                                     $if (!((hit->inst == source_instance) &
                                            (hit->prim == source_primitive))) {
                                         candidate.commit();
                                     };
                                 })
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();
        results.write(0u,
                      make_float4(committed->committed_ray_t,
                                  cast<float>(committed->prim),
                                  cast<float>(committed->hit_type),
                                  cast<float>(callback_count)));
    };

    auto shader = device.compile(trace);
    std::array<float4, 1u> host_results{};
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << shader(accel, result_buffer).dispatch(1u)
           << result_buffer.copy_to(luisa::span{host_results})
           << synchronize();

    expect_near(host_results[0].x, 2.4811954f, "closest committed distance");
    expect(static_cast<uint>(host_results[0].y) < 2u)
        << luisa::format("closest committed primitive: got {}, expected near quad",
                         host_results[0].y);
    expect(static_cast<uint>(host_results[0].z) ==
           static_cast<uint>(HitType::Surface));
    expect(host_results[0].w >= 1.0f);
}

void test_inline_ray_query_dsl(Device &device) {
    auto stream = device.create_stream();

    const std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(
        mesh, make_float4x4(1.0f), 0xffu, false);
    accel.emplace_back(
        mesh, translation(make_float3(0.0f, 0.0f, -2.0f)),
        0xffu, false);

    constexpr auto dispatch_size = 5u;
    auto result_buffer = device.create_buffer<uint4>(dispatch_size);
    Kernel1D trace = [](AccelVar accel, BufferUInt4 results) noexcept {
        auto index = dispatch_x();
        auto target = index & 1u;
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 10.0f);
        auto query = accel.query(ray, {});
        UInt callback_count = 0u;
        $while (query.proceed()) {
            $if (query.is_surface_candidate()) {
                auto candidate = query.surface_candidate();
                callback_count += 1u;
                $if (candidate.hit()->inst == target) {
                    candidate.commit();
                };
            }
            $else {
                // The scene has no procedural geometry. Keeping the explicit
                // arm exercises the portable frontend normalization shape.
                auto candidate = query.procedural_candidate();
                $if (candidate.hit()->inst == ~0u) {
                    candidate.terminate();
                };
            };
        };
        auto committed = query.committed_hit();
        results.write(
            index,
            make_uint4(
                committed->hit_type, committed->inst,
                committed->prim, callback_count));
    };

    auto shader = device.compile(trace);
    std::array<uint4, dispatch_size> host_results{};
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << shader(accel, result_buffer).dispatch(dispatch_size)
           << result_buffer.copy_to(luisa::span{host_results})
           << synchronize();

    for (auto i = 0u; i < host_results.size(); i++) {
        auto target = i & 1u;
        const auto result = host_results[i];
        // Traversal may visit the far candidate before the near one. A near
        // commit can cull the far candidate; a far commit still visits both.
        auto valid_count = target == 0u ? result.w >= 1u && result.w <= 2u : result.w == 2u;
        expect(result.x == static_cast<uint32_t>(HitType::Surface) &&
               result.y == target && result.z == 0u && valid_count)
            << luisa::format(
                   "inline ray-query mismatch at lane {}: got ({}, {}, {}, {})",
                   i, host_results[i].x, host_results[i].y,
                   host_results[i].z, host_results[i].w);
    }
}

void test_stateful_inline_ray_query_dsl(Device &device) {
    auto stream = device.create_stream();

    const std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    const std::array bounds{
        AABB{.packed_min = {-0.5f, -0.5f, -0.1f},
             .packed_max = {0.5f, 0.5f, 0.1f}}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto bounds_buffer = device.create_buffer<AABB>(bounds.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto procedural = device.create_procedural_primitive(bounds_buffer);

    // 1. Triangle-only `$while (rq.proceed())` without `is_surface_candidate()` split.
    auto tri_accel = device.create_accel();
    tri_accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    tri_accel.emplace_back(mesh, translation(make_float3(0.0f, 0.0f, -2.0f)), 0xffu, false);

    constexpr auto dispatch_size = 4u;
    auto result_buffer = device.create_buffer<uint4>(dispatch_size);
    Kernel1D tri_only_kernel = [](AccelVar accel, BufferUInt4 results) noexcept {
        auto index = dispatch_x();
        auto target = index & 1u;
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 10.0f);
        auto rq = accel.query(ray, {});
        UInt visited = 0u;
        $while (rq.proceed()) {
            visited += 1u;
            auto candidate = rq.surface_candidate();
            $if (candidate.hit()->inst == target) {
                candidate.commit();
            };
        };
        auto hit = rq.committed_hit();
        results.write(index, make_uint4(hit->hit_type, hit->inst, hit->prim, visited));
    };

    // 2. Mixed surface+procedural manual `$loop` with shared pre/post state, stored predicate, `$continue`, and `$break`.
    auto mixed_accel = device.create_accel();
    mixed_accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    mixed_accel.emplace_back(procedural, translation(make_float3(0.0f, 0.0f, -2.0f)), 0xffu);

    auto mixed_result_buffer = device.create_buffer<uint4>(dispatch_size);
    Kernel1D mixed_stateful_kernel = [](AccelVar accel, BufferUInt4 results) noexcept {
        auto index = dispatch_x();
        auto commit_surface = (index & 1u) == 0u;
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 10.0f);
        auto rq = accel.query(ray, {});
        UInt visited = 0u;
        UInt post = 0u;
        $loop {
            $if (!rq.proceed()) { $break; };
            visited += 1u;
            Bool is_surf = rq.is_surface_candidate();
            $if (!is_surf) {
                auto proc = rq.procedural_candidate();
                $if (!commit_surface) {
                    proc.commit(3.0f);
                };
                $continue;
            };
            auto surf = rq.surface_candidate();
            $if (commit_surface) {
                surf.commit();
                $break;
            };
            post += 1u;
        };
        auto hit = rq.committed_hit();
        results.write(index, make_uint4(hit->hit_type, hit->inst, visited, post));
    };

    auto tri_shader = device.compile(tri_only_kernel);
    auto mixed_shader = device.compile(mixed_stateful_kernel);
    std::array<uint4, dispatch_size> tri_results{};
    std::array<uint4, dispatch_size> mixed_results{};
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << bounds_buffer.copy_from(luisa::span{bounds})
           << mesh.build()
           << procedural.build()
           << tri_accel.build()
           << mixed_accel.build()
           << tri_shader(tri_accel, result_buffer).dispatch(dispatch_size)
           << mixed_shader(mixed_accel, mixed_result_buffer).dispatch(dispatch_size)
           << result_buffer.copy_to(luisa::span{tri_results})
           << mixed_result_buffer.copy_to(luisa::span{mixed_results})
           << synchronize();

    for (auto i = 0u; i < dispatch_size; i++) {
        auto target = i & 1u;
        const auto r = tri_results[i];
        auto valid_visited = target == 0u ? (r.w >= 1u && r.w <= 2u) : (r.w == 2u);
        expect(r.x == static_cast<uint32_t>(HitType::Surface) &&
               r.y == target && r.z == 0u && valid_visited)
            << luisa::format("triangle-only inline ray-query mismatch at lane {}: ({}, {}, {}, {})",
                              i, r.x, r.y, r.z, r.w);

        const auto m = mixed_results[i];
        if ((i & 1u) == 0u) {
            expect(m.x == static_cast<uint32_t>(HitType::Surface) &&
                   m.y == 0u && m.z >= 1u && m.z <= 2u && m.w == 0u)
                << luisa::format("mixed stateful inline ray-query (surface) mismatch at lane {}: ({}, {}, {}, {})",
                                  i, m.x, m.y, m.z, m.w);
        } else {
            expect(m.x == static_cast<uint32_t>(HitType::Procedural) &&
                   m.y == 1u && m.z == 2u && m.w == 1u)
                << luisa::format("mixed stateful inline ray-query (procedural) mismatch at lane {}: ({}, {}, {}, {})",
                                  i, m.x, m.y, m.z, m.w);
        }
    }
}

void test_negative_surface_barycentric_hit_type(Device &device) {
    auto stream = device.create_stream();

    // Reduced from a real shared-edge hit in Blender's Monster Under the Bed
    // scene. Embree's watertight intersector can legitimately report a tiny
    // negative barycentric coordinate for the triangle that owns the edge.
    // Hit identity must therefore be carried independently of barycentrics.
    const std::array vertices{
        make_float3(luisa::bit_cast<float>(0x3fc545ffu),
                    luisa::bit_cast<float>(0x3ddc59cbu),
                    luisa::bit_cast<float>(0x3f68f44fu)),
        make_float3(luisa::bit_cast<float>(0x3fc56935u),
                    luisa::bit_cast<float>(0x3df097c8u),
                    luisa::bit_cast<float>(0x3f66331eu)),
        make_float3(luisa::bit_cast<float>(0x3fc3cad4u),
                    luisa::bit_cast<float>(0x3e020697u),
                    luisa::bit_cast<float>(0x3f67c6feu)),
        make_float3(luisa::bit_cast<float>(0x3fc38515u),
                    luisa::bit_cast<float>(0x3ded2a7eu),
                    luisa::bit_cast<float>(0x3f6a9e9au))};
    const std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{0u, 2u, 3u}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, true);

    auto result_buffer = device.create_buffer<float4>(1u);
    Kernel1D trace = [](AccelVar accel, BufferFloat4 results) noexcept {
        auto ray = make_ray(
            make_float3(0x1.889388p+0f, 0x1.afaaa4p-4f,
                        0x1.c64612p-1f),
            make_float3(0x1.7312ep-5f, 0x1.19d2e4p-1f,
                        0x1.aad3c4p-1f),
            0.0f, 0x1.0a6f96p-5f);
        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate(
                                 [](SurfaceCandidate &) noexcept {})
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();
        results.write(0u, make_float4(
                              cast<float>(committed->hit_type),
                              committed->bary.x, committed->bary.y,
                              committed->committed_ray_t));
    };

    auto shader = device.compile(trace);
    std::array<float4, 1u> host_results{};
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << shader(accel, result_buffer).dispatch(1u)
           << result_buffer.copy_to(luisa::span{host_results})
           << synchronize();

    expect(static_cast<uint>(host_results[0].x) ==
           static_cast<uint>(HitType::Surface))
        << luisa::format(
               "shared-edge hit type: got {}, bary=({}, {}), t={}",
               host_results[0].x, host_results[0].y,
               host_results[0].z, host_results[0].w);
}

void test_procedural_ray_query_world_ray(Device &device) {
    auto stream = device.create_stream();

    const std::array bounds{
        AABB{
            .packed_min = {-0.5f, -0.5f, -0.1f},
            .packed_max = {0.5f, 0.5f, 0.1f}}};
    auto bounds_buffer =
        device.create_buffer<AABB>(bounds.size());
    auto procedural =
        device.create_procedural_primitive(bounds_buffer);
    auto accel = device.create_accel();
    const auto transform = make_float4x4(
        make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        make_float4(0.0f, 1.0f, 0.0f, 0.0f),
        make_float4(0.0f, 0.0f, 2.0f, 0.0f),
        make_float4(0.0f, 0.0f, 1.0f, 1.0f));
    accel.emplace_back(procedural, transform, 0xffu);

    auto result_buffer = device.create_buffer<float4>(5u);
    Kernel1D trace = [](AccelVar accel,
                        BufferFloat4 results) noexcept {
        constexpr auto initial_origin =
            make_float3(0.125f, -0.125f, 3.0f);
        constexpr auto initial_direction =
            make_float3(0.0f, 0.0f, -2.0f);
        auto ray = make_ray(
            initial_origin,
            initial_direction,
            0.25f,
            7.5f);
        UInt callback_count = 0u;
        auto committed =
            accel.traverse(ray, {})
                .on_surface_candidate(
                    [](SurfaceCandidate &) noexcept {})
                .on_procedural_candidate(
                    [&](ProceduralCandidate
                            &candidate) noexcept {
                        auto before = candidate.ray();
                        results.write(
                            0u,
                            make_float4(
                                before->origin(),
                                before->t_min()));
                        results.write(
                            1u,
                            make_float4(
                                before->direction(),
                                before->t_max()));
                        callback_count += 1u;
                        candidate.commit(1.0f);
                        auto after = candidate.ray();
                        results.write(
                            2u,
                            make_float4(
                                after->origin(),
                                after->t_min()));
                        results.write(
                            3u,
                            make_float4(
                                after->direction(),
                                after->t_max()));
                    })
                .trace();
        results.write(
            4u,
            make_float4(
                committed->distance(),
                cast<float>(committed->hit_type),
                cast<float>(callback_count),
                0.0f));
    };

    auto shader = device.compile(trace);
    std::array<float4, 5u> host_results{};
    stream
        << bounds_buffer.copy_from(luisa::span{bounds})
        << procedural.build()
        << accel.build()
        << shader(accel, result_buffer).dispatch(1u)
        << result_buffer.copy_to(
               luisa::span{host_results})
        << synchronize();

    auto check_immutable_ray_fields =
        [&](const float4 &origin_tmin,
            const float4 &direction_tmax,
            luisa::string_view phase) noexcept {
            expect_near(
                origin_tmin.x,
                0.125f,
                luisa::format(
                    "{} origin.x", phase));
            expect_near(
                origin_tmin.y,
                -0.125f,
                luisa::format(
                    "{} origin.y", phase));
            expect_near(
                origin_tmin.z,
                3.0f,
                luisa::format(
                    "{} origin.z", phase));
            expect_near(
                origin_tmin.w,
                0.25f,
                luisa::format(
                    "{} t_min", phase));
            expect_near(
                direction_tmax.x,
                0.0f,
                luisa::format(
                    "{} direction.x", phase));
            expect_near(
                direction_tmax.y,
                0.0f,
                luisa::format(
                    "{} direction.y", phase));
            expect_near(
                direction_tmax.z,
                -2.0f,
                luisa::format(
                    "{} direction.z", phase));
        };
    check_immutable_ray_fields(
        host_results[0],
        host_results[1],
        "procedural before commit");
    check_immutable_ray_fields(
        host_results[2],
        host_results[3],
        "procedural after commit");
    expect_near(
        host_results[1].w,
        7.5f,
        "procedural pre-commit t_max");
    expect_near(
        host_results[3].w,
        1.0f,
        "procedural post-commit t_max");
    expect_near(
        host_results[4].x,
        1.0f,
        "procedural committed distance");
    expect(
        static_cast<uint>(host_results[4].y) ==
        static_cast<uint>(HitType::Procedural));
    expect(
        static_cast<uint>(host_results[4].z) == 1u);

    // Compile a distinct object-ray-only handler. Besides making the contract
    // explicit, this guards the HIP native-closest quotient: the callback must
    // consume HIPRT's already-transformed active ray without materializing an
    // instance matrix in user code or aliasing the world-ray representation.
    Kernel1D trace_object_ray = [](
                                    AccelVar accel,
                                    BufferFloat4 results) noexcept {
        auto ray = make_ray(
            make_float3(0.125f, -0.125f, 3.0f),
            make_float3(0.0f, 0.0f, -2.0f),
            0.25f, 7.5f);
        UInt callback_count = 0u;
        auto committed =
            accel.traverse(ray, {})
                .on_surface_candidate(
                    [](SurfaceCandidate &) noexcept {})
                .on_procedural_candidate(
                    [&](ProceduralCandidate &candidate) noexcept {
                        auto before = candidate.object_ray();
                        results.write(
                            0u, make_float4(
                                    before->origin(), before->t_min()));
                        results.write(
                            1u, make_float4(
                                    before->direction(), before->t_max()));
                        callback_count += 1u;
                        candidate.commit(1.0f);
                        auto after = candidate.object_ray();
                        results.write(
                            2u, make_float4(
                                    after->origin(), after->t_min()));
                        results.write(
                            3u, make_float4(
                                    after->direction(), after->t_max()));
                    })
                .trace();
        results.write(
            4u, make_float4(
                    committed->distance(),
                    cast<float>(committed->hit_type),
                    cast<float>(callback_count), 0.0f));
    };
    auto object_ray_shader = device.compile(trace_object_ray);
    host_results.fill({});
    stream << object_ray_shader(accel, result_buffer).dispatch(1u)
           << result_buffer.copy_to(luisa::span{host_results})
           << synchronize();

    auto check_object_ray_fields =
        [&](const float4 &origin_tmin,
            const float4 &direction_tmax,
            luisa::string_view phase) noexcept {
            expect_near(origin_tmin.x, 0.125f,
                        luisa::format("{} origin.x", phase));
            expect_near(origin_tmin.y, -0.125f,
                        luisa::format("{} origin.y", phase));
            expect_near(origin_tmin.z, 1.0f,
                        luisa::format("{} origin.z", phase));
            expect_near(origin_tmin.w, 0.25f,
                        luisa::format("{} t_min", phase));
            expect_near(direction_tmax.x, 0.0f,
                        luisa::format("{} direction.x", phase));
            expect_near(direction_tmax.y, 0.0f,
                        luisa::format("{} direction.y", phase));
            expect_near(direction_tmax.z, -1.0f,
                        luisa::format("{} direction.z", phase));
        };
    check_object_ray_fields(
        host_results[0], host_results[1],
        "procedural object ray before commit");
    check_object_ray_fields(
        host_results[2], host_results[3],
        "procedural object ray after commit");
    expect_near(host_results[1].w, 7.5f,
                "procedural object pre-commit t_max");
    expect_near(host_results[3].w, 1.0f,
                "procedural object post-commit t_max");
    expect_near(host_results[4].x, 1.0f,
                "procedural object committed distance");
    expect(static_cast<uint>(host_results[4].y) ==
           static_cast<uint>(HitType::Procedural));
    expect(static_cast<uint>(host_results[4].z) == 1u);

    // A handler may observe both representations. Their product cannot be
    // encoded by aliasing one mutable ray field: the world ray is immutable,
    // while the object ray is candidate-dependent. HIP therefore classifies
    // the containing function into the exact resumable state domain.
    Kernel1D trace_both_rays = [](
                                   AccelVar accel,
                                   BufferFloat4 results) noexcept {
        auto ray = make_ray(
            make_float3(0.125f, -0.125f, 3.0f),
            make_float3(0.0f, 0.0f, -2.0f),
            0.25f, 7.5f);
        UInt callback_count = 0u;
        auto committed =
            accel.traverse(ray, {})
                .on_surface_candidate(
                    [](SurfaceCandidate &) noexcept {})
                .on_procedural_candidate(
                    [&](ProceduralCandidate &candidate) noexcept {
                        auto world = candidate.ray();
                        auto object = candidate.object_ray();
                        results.write(
                            0u, make_float4(
                                    world->origin(), world->t_min()));
                        results.write(
                            1u, make_float4(
                                    object->origin(), object->t_min()));
                        results.write(
                            2u, make_float4(
                                    world->direction(), world->t_max()));
                        results.write(
                            3u, make_float4(
                                    object->direction(), object->t_max()));
                        callback_count += 1u;
                        candidate.commit(1.0f);
                        results.write(
                            4u, make_float4(
                                    candidate.ray()->t_max(),
                                    candidate.object_ray()->t_max(),
                                    cast<float>(callback_count), 0.0f));
                    })
                .trace();
        results.atomic(4u).w.fetch_add(
            committed->distance());
    };
    auto both_rays_shader = device.compile(trace_both_rays);
    host_results.fill({});
    stream << both_rays_shader(accel, result_buffer).dispatch(1u)
           << result_buffer.copy_to(luisa::span{host_results})
           << synchronize();

    check_immutable_ray_fields(
        host_results[0], host_results[2],
        "joint world/object world ray");
    check_object_ray_fields(
        host_results[1], host_results[3],
        "joint world/object object ray");
    expect_near(host_results[2].w, 7.5f,
                "joint world/object world pre-commit t_max");
    expect_near(host_results[3].w, 7.5f,
                "joint world/object object pre-commit t_max");
    expect_near(host_results[4].x, 1.0f,
                "joint world/object world post-commit t_max");
    expect_near(host_results[4].y, 1.0f,
                "joint world/object object post-commit t_max");
    expect_near(host_results[4].z, 1.0f,
                "joint world/object callback count");
    expect_near(host_results[4].w, 1.0f,
                "joint world/object committed distance");
}

void test_ray_query_world_ray_opaque_and_non_opaque(Device &device) {
    auto stream = device.create_stream();

    const std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);

    // Instance 0 (non-opaque): z = 1.5 -> t = 0.5
    // Instance 1 (opaque):     z = 1.0 -> t = 1.0
    auto accel = device.create_accel();
    accel.emplace_back(mesh, translation(make_float3(0.0f, 0.0f, 1.5f)), 0xffu, false);
    accel.emplace_back(mesh, translation(make_float3(0.0f, 0.0f, 1.0f)), 0xffu, true);

    auto results_buf = device.create_buffer<float4>(2u);
    Kernel1D trace = [](AccelVar accel, UInt commit_non_opaque, BufferFloat4 results) noexcept {
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 2.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 7.5f);
        Float world_pre = -1.0f;
        Float object_pre = -1.0f;
        Float world_post = -1.0f;
        Float object_post = -1.0f;
        UInt callback_count = 0u;
        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                 callback_count += 1u;
                                 world_pre = candidate.ray()->t_max();
                                 object_pre = candidate.object_ray()->t_max();
                                 $if (commit_non_opaque != 0u) {
                                     candidate.commit();
                                 };
                                 world_post = candidate.ray()->t_max();
                                 object_post = candidate.object_ray()->t_max();
                             })
                             .on_procedural_candidate([](ProceduralCandidate &) noexcept {})
                             .trace();
        results.write(0u, make_float4(world_pre, object_pre, world_post, object_post));
        results.write(1u, make_float4(committed->distance(),
                                      cast<float>(committed->inst),
                                      cast<float>(committed->hit_type),
                                      cast<float>(callback_count)));
    };

    auto shader = device.compile(trace);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    std::array<float4, 2u> host_results{};
    // Reject non-opaque (t=0.5), fall back to opaque (t=1.0).
    stream << shader(accel, 0u, results_buf).dispatch(1u)
           << results_buf.copy_to(luisa::span{host_results})
           << synchronize();
    expect(host_results[0].x >= 1.0f - tolerance && host_results[0].x <= 7.5f + tolerance)
        << luisa::format("mixed opaque/non-opaque reject pre world t_max={}", host_results[0].x);
    expect(host_results[0].y >= 1.0f - tolerance && host_results[0].y <= 7.5f + tolerance)
        << luisa::format("mixed opaque/non-opaque reject pre object t_max={}", host_results[0].y);
    expect_near(host_results[0].z, host_results[0].x, "mixed reject post world t_max");
    expect_near(host_results[0].w, host_results[0].y, "mixed reject post object t_max");
    expect_near(host_results[1].x, 1.0f, "mixed reject committed distance");
    expect_near(host_results[1].y, 1.0f, "mixed reject committed instance");

    // Commit non-opaque (t=0.5), beating opaque (t=1.0).
    stream << shader(accel, 1u, results_buf).dispatch(1u)
           << results_buf.copy_to(luisa::span{host_results})
           << synchronize();
    expect_near(host_results[0].z, 0.5f, "mixed commit post world t_max");
    expect_near(host_results[0].w, 0.5f, "mixed commit post object t_max");
    expect_near(host_results[1].x, 0.5f, "mixed commit committed distance");
    expect_near(host_results[1].y, 0.0f, "mixed commit committed instance");
}

void test_stateful_inline_ray_query_corner_cases(Device &device) {
    auto stream = device.create_stream();

    const std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    const std::array bounds{
        AABB{.packed_min = {-0.5f, -0.5f, -0.1f},
             .packed_max = {0.5f, 0.5f, 0.1f}}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto bounds_buffer = device.create_buffer<AABB>(bounds.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto procedural = device.create_procedural_primitive(bounds_buffer);

    // Non-opaque surface at z=0.0 (t=1.0 from z=1.0), opaque surface at z=-1.0 (t=2.0),
    // procedural at z=-2.0 (t=3.0).
    auto mixed_accel = device.create_accel();
    mixed_accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    mixed_accel.emplace_back(mesh, translation(make_float3(0.0f, 0.0f, -1.0f)), 0xffu, false);
    mixed_accel.emplace_back(procedural, translation(make_float3(0.0f, 0.0f, -2.0f)), 0xffu);

    auto tri_only_accel = device.create_accel();
    tri_only_accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);

    auto img = device.create_image<float>(PixelStorage::FLOAT4, 4u, 1u);
    const std::array<float4, 4u> img_init{
        make_float4(1.0f, 2.0f, 3.0f, 4.0f),
        make_float4(5.0f, 6.0f, 7.0f, 8.0f),
        make_float4(9.0f, 10.0f, 11.0f, 12.0f),
        make_float4(13.0f, 14.0f, 15.0f, 16.0f)};

    constexpr auto dispatch_size = 4u;
    auto extra_buf = device.create_buffer<uint>(4u);
    const std::array<uint, 4u> extra_init{10u, 20u, 30u, 40u};
    auto results_u = device.create_buffer<uint4>(dispatch_size * 4u);
    auto results_f = device.create_buffer<float4>(dispatch_size * 2u);

    // 1. $while (rq.proceed()) inside a custom Callable.
    Callable callable_query = [](AccelVar accel, UInt target) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
        auto rq = accel.query(ray, {});
        UInt count = 0u;
        $while (rq.proceed()) {
            count += 1u;
            $if (rq.is_surface_candidate()) {
                $if (rq.surface_candidate().hit()->inst == target) {
                    rq.surface_candidate().commit();
                };
            };
        };
        auto h = rq.committed_hit();
        return make_uint2(h->inst, count);
    };

    Kernel1D corner_kernel = [&callable_query](AccelVar mixed, AccelVar tri_only,
                                               BufferUInt buf, ImageFloat tex,
                                               BufferUInt4 out_u, BufferFloat4 out_f) noexcept {
        auto idx = dispatch_x();
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.1f, 8.0f);

        // A. Custom Callable + outer $for + sequential loops + empty loop.
        auto from_callable = callable_query(mixed, idx & 1u);

        auto empty_q = tri_only.query(ray, {});
        $while (empty_q.proceed()) {};

        UInt seq_sum = 0u;
        for (auto iter : dynamic_range(2u)) {
            auto loop_q = tri_only.query(ray, {});
            $while (loop_q.proceed()) {
                $if (loop_q.is_surface_candidate()) {
                    loop_q.surface_candidate().commit();
                };
            };
            seq_sum += iter + loop_q.committed_hit()->inst;
        }

        // B. Inner $for + inner $switch + pre-declared Bool + nested-block Bool + compound predicates
        //    + Buffer/Image/large ArrayUInt<40> captures + ray()/object_ray() t_max.
        ArrayUInt<40u> big_arr;
        $for (k, 40u) {
            big_arr[k] = idx + k;
        };
        Float4 tex_val = make_float4(0.0f);
        UInt buf_val = 0u;
        UInt sw_acc = 0u;
        Float pre_w_tmax = -1.0f;
        Float post_w_tmax = -1.0f;
        Float post_o_tmax = -1.0f;
        UInt inside_commit_type = 0u;
        Bool pre_is_surf = false;
        auto complex_q = mixed.query(ray, {});
        $while (complex_q.proceed()) {
            pre_is_surf = complex_q.is_surface_candidate();
            for (auto s : dynamic_range(2u)) {
                $if (s == 0u) { $continue; };
                sw_acc += 1u;
                $break;
            }
            Bool do_outer_break = false;
            $if (pre_is_surf & (idx < 100u)) {
                auto sc = complex_q.surface_candidate();
                $switch (sc.hit()->inst) {
                    $case (0u) {
                        $if ((idx & 1u) == 0u) {
                            pre_w_tmax = sc.ray()->t_max();
                            sc.commit();
                            post_w_tmax = sc.ray()->t_max();
                            post_o_tmax = sc.object_ray()->t_max();
                            tex_val = tex.read(make_uint2(idx, 0u));
                            buf_val = buf.read(idx);
                            big_arr[idx + 5u] += 100u;
                            inside_commit_type = complex_q.committed_hit()->hit_type;
                            do_outer_break = true;
                            $break;
                        };
                    };
                    $default {
                        $if ((idx & 1u) == 1u) {
                            pre_w_tmax = sc.ray()->t_max();
                            sc.commit();
                            post_w_tmax = sc.ray()->t_max();
                            post_o_tmax = sc.object_ray()->t_max();
                            tex_val = tex.read(make_uint2(idx, 0u));
                            buf_val = buf.read(idx);
                            big_arr[idx + 5u] += 200u;
                            inside_commit_type = complex_q.committed_hit()->hit_type;
                        };
                    };
                };
                $if (do_outer_break) { $break; };
            }
            $else {
                $if (idx == idx) {
                    Bool nested_flag = complex_q.is_surface_candidate();
                    Bool chosen = ite(nested_flag, idx > 1000u, idx < 1000u);
                    $if (!nested_flag & chosen) {
                        auto pc = complex_q.procedural_candidate();
                        $if ((idx & 1u) == 0u) {
                            pc.commit(3.0f);
                        };
                    };
                };
            };
        };
        auto c_hit = complex_q.committed_hit();

        // C. query_any termination check.
        auto any_q = mixed.query_any(ray, {});
        UInt any_calls = 0u;
        $while (any_q.proceed()) {
            any_calls += 1u;
            $if (any_q.is_surface_candidate()) {
                any_q.surface_candidate().commit();
            }
            $else {
                any_q.procedural_candidate().commit(3.0f);
            };
        };

        out_u.write(idx * 4u + 0u, make_uint4(from_callable.x, from_callable.y,
                                              empty_q.committed_hit()->hit_type, seq_sum));
        out_u.write(idx * 4u + 1u, make_uint4(c_hit->hit_type, c_hit->inst, sw_acc, big_arr[idx + 5u]));
        out_u.write(idx * 4u + 2u, make_uint4(buf_val, inside_commit_type,
                                              any_q.committed_hit()->hit_type, any_calls));
        out_f.write(idx * 2u + 0u, tex_val);
        out_f.write(idx * 2u + 1u, make_float4(pre_w_tmax, post_w_tmax, post_o_tmax, c_hit->distance()));
    };

    // Motion blur inline stateful loop + (LLVM-only) uncommitted $break preservation.
    auto llvm_env = std::getenv("LUISA_EXPERIMENTAL_LLVM_CODEGEN");
    auto is_cuda_llvm = llvm_env != nullptr && std::strcmp(llvm_env, "1") == 0;
    auto is_cuda = device.backend_name() == "cuda";
    AccelMotionOption motion_opt{};
    motion_opt.mode = AccelMotionMode::MATRIX;
    motion_opt.keyframe_count = 2u;
    motion_opt.time_start = 0.0f;
    motion_opt.time_end = 1.0f;
    auto motion_inst = device.create_motion_instance(mesh, motion_opt);
    const std::array<float4x4, 2u> motion_keys{
        make_float4x4(1.0f),
        translation(make_float3(0.0f, 0.0f, -2.0f))};
    motion_inst.set_keyframes(luisa::span{motion_keys});
    auto motion_accel = device.create_accel();
    motion_accel.emplace_back(motion_inst, make_float4x4(1.0f), 0xffu, false);
    auto motion_res = device.create_buffer<float4>(dispatch_size);

    Kernel1D motion_kernel = [](AccelVar motion_scene, BufferFloat4 out) noexcept {
        auto idx = dispatch_x();
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
        auto mq = motion_scene.query_motion(ray, 0.5f, {});
        UInt count = 0u;
        $while (mq.proceed()) {
            count += 1u;
            $if (mq.is_surface_candidate()) {
                mq.surface_candidate().commit();
            };
        };
        auto mh = mq.committed_hit();
        out.write(idx, make_float4(cast<float>(mh->hit_type), mh->distance(),
                                    cast<float>(mh->inst), cast<float>(count)));
    };

    auto shader = device.compile(corner_kernel);
    auto motion_shader = device.compile(motion_kernel);

    std::array<uint4, dispatch_size * 4u> host_u{};
    std::array<float4, dispatch_size * 2u> host_f{};
    std::array<float4, dispatch_size> host_motion{};
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << bounds_buffer.copy_from(luisa::span{bounds})
           << img.copy_from(img_init.data())
           << extra_buf.copy_from(luisa::span{extra_init})
           << mesh.build()
           << procedural.build()
           << motion_inst.build()
           << mixed_accel.build()
           << tri_only_accel.build()
           << motion_accel.build()
           << shader(mixed_accel, tri_only_accel, extra_buf, img, results_u, results_f).dispatch(dispatch_size)
           << motion_shader(motion_accel, motion_res).dispatch(dispatch_size)
           << results_u.copy_to(luisa::span{host_u})
           << results_f.copy_to(luisa::span{host_f})
           << motion_res.copy_to(luisa::span{host_motion})
           << synchronize();

    for (auto i = 0u; i < dispatch_size; i++) {
        auto u0 = host_u[i * 4u + 0u];
        auto u1 = host_u[i * 4u + 1u];
        auto u2 = host_u[i * 4u + 2u];
        auto f0 = host_f[i * 2u + 0u];
        auto f1 = host_f[i * 2u + 1u];
        auto m = host_motion[i];

        expect(u0.x == (i & 1u) && u0.y >= 1u &&
               u0.z == static_cast<uint>(HitType::Miss) && u0.w == 1u)
            << luisa::format("stateful corner callable/empty/seq mismatch lane {}: ({},{},{},{})",
                             i, u0.x, u0.y, u0.z, u0.w);

        if ((i & 1u) == 0u) {
            expect(u1.x == static_cast<uint>(HitType::Surface) && u1.y == 0u &&
                   u1.z >= 1u && u1.w == (i * 2u + 5u + 100u))
                << luisa::format("stateful corner even complex mismatch lane {}: ({},{},{},{})",
                                 i, u1.x, u1.y, u1.z, u1.w);
            expect_near(f1.y, 1.0f, "stateful corner even post world t_max");
            expect_near(f1.z, 1.0f, "stateful corner even post object t_max");
            expect_near(f1.w, 1.0f, "stateful corner even committed t");
        } else {
            expect(u1.x == static_cast<uint>(HitType::Surface) && u1.y == 1u &&
                   u1.z >= 1u && u1.w == (i * 2u + 5u + 200u))
                << luisa::format("stateful corner odd complex mismatch lane {}: ({},{},{},{})",
                                 i, u1.x, u1.y, u1.z, u1.w);
            expect_near(f1.y, 2.0f, "stateful corner odd post world t_max");
            expect_near(f1.z, 2.0f, "stateful corner odd post object t_max");
            expect_near(f1.w, 2.0f, "stateful corner odd committed t");
        }
        expect(u2.x == extra_init[i] &&
               u2.z == static_cast<uint>(HitType::Surface) && u2.w == 1u)
            << luisa::format("stateful corner capture/any mismatch lane {}: ({},{},{},{})",
                             i, u2.x, u2.y, u2.z, u2.w);
        expect(f1.x >= 1.0f - tolerance && f1.x <= 8.0f + tolerance)
            << luisa::format("stateful corner pre t_max invalid lane {}: {}", i, f1.x);
        expect_near(f0.x, img_init[i].x, "stateful corner image capture x");
        expect_near(f0.w, img_init[i].w, "stateful corner image capture w");
        expect_near(m.x, static_cast<float>(HitType::Surface), "stateful corner motion hit_type");
        expect_near(m.y, 2.0f, "stateful corner motion t=0.5 distance");
        expect_near(m.z, 0.0f, "stateful corner motion inst");
        expect_near(m.w, 1.0f, "stateful corner motion callback count");
    }

    // Exact uncommitted-$break preservation requires software query state inside AH/IS
    // (guaranteed on CUDA LLVM or non-CUDA-AST backends). Use two overlapping procedural
    // AABBs and commit the farther t on the first visit so hardware still visits the second.
    if (!is_cuda || is_cuda_llvm) {
        auto break_accel = device.create_accel();
        break_accel.emplace_back(procedural, translation(make_float3(0.0f, 0.0f, -1.0f)), 0xffu);
        break_accel.emplace_back(procedural, translation(make_float3(0.0f, 0.0f, -1.0f)), 0xffu);
        auto break_res = device.create_buffer<uint4>(1u);
        Kernel1D break_preserve = [](AccelVar accel, BufferUInt4 out) noexcept {
            auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                                make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
            auto rq = accel.query(ray, {});
            UInt calls = 0u;
            $while (rq.proceed()) {
                calls += 1u;
                $if (rq.is_procedural_candidate()) {
                    $if (calls == 1u) {
                        rq.procedural_candidate().commit(2.05f);
                    }
                    $else {
                        $break;
                    };
                };
            };
            auto h = rq.committed_hit();
            out.write(0u, make_uint4(h->hit_type, h->inst, calls, as<uint>(h->distance())));
        };
        auto break_shader = device.compile(break_preserve);
        std::array<uint4, 1u> host_break{};
        stream << break_accel.build()
               << break_shader(break_accel, break_res).dispatch(1u)
               << break_res.copy_to(luisa::span{host_break})
               << synchronize();
        expect(host_break[0].x == static_cast<uint>(HitType::Procedural) &&
               host_break[0].z == 2u &&
               std::abs(std::bit_cast<float>(host_break[0].w) - 2.05f) < tolerance)
            << luisa::format("uncommitted break preserved hit mismatch: ({},{},{},{})",
                             host_break[0].x, host_break[0].y,
                             host_break[0].z, host_break[0].w);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_ray_query_world_ray(dc->device);
    test_ray_query_commits_closest_surface(dc->device);
    test_inline_ray_query_dsl(dc->device);
    test_stateful_inline_ray_query_dsl(dc->device);
    test_negative_surface_barycentric_hit_type(dc->device);
    test_procedural_ray_query_world_ray(dc->device);
    test_ray_query_world_ray_opaque_and_non_opaque(dc->device);
    test_stateful_inline_ray_query_corner_cases(dc->device);
}
