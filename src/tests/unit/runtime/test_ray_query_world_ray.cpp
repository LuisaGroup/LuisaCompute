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
                                     $if(!((hit->inst == source_instance) &
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
        expect(static_cast<bool>(
            all(host_results[i] ==
                make_uint4(
                    static_cast<uint32_t>(HitType::Surface),
                    target, 0u, target + 1u))))
            << luisa::format(
                   "inline ray-query mismatch at lane {}: got ({}, {}, {}, {})",
                   i, host_results[i].x, host_results[i].y,
                   host_results[i].z, host_results[i].w);
    }
}

void test_negative_surface_barycentric_hit_type(Device &device) {
    auto stream = device.create_stream();

    // Reduced from a real shared-edge hit in Blender's Monster Under the Bed
    // scene. Embree's watertight intersector can legitimately report a tiny
    // negative barycentric coordinate for the triangle that owns the edge.
    // Hit identity must therefore be carried independently of barycentrics.
    const std::array vertices{
        make_float3(std::bit_cast<float>(0x3fc545ffu),
                    std::bit_cast<float>(0x3ddc59cbu),
                    std::bit_cast<float>(0x3f68f44fu)),
        make_float3(std::bit_cast<float>(0x3fc56935u),
                    std::bit_cast<float>(0x3df097c8u),
                    std::bit_cast<float>(0x3f66331eu)),
        make_float3(std::bit_cast<float>(0x3fc3cad4u),
                    std::bit_cast<float>(0x3e020697u),
                    std::bit_cast<float>(0x3f67c6feu)),
        make_float3(std::bit_cast<float>(0x3fc38515u),
                    std::bit_cast<float>(0x3ded2a7eu),
                    std::bit_cast<float>(0x3f6a9e9au))};
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

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_ray_query_world_ray(dc->device);
    test_ray_query_commits_closest_surface(dc->device);
    test_inline_ray_query_dsl(dc->device);
    test_negative_surface_barycentric_hit_type(dc->device);
    test_procedural_ray_query_world_ray(dc->device);
}
