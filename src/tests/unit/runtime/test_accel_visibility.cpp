// Test for acceleration-structure instance visibility masks.
// This test covers:
// - closest-hit, any-hit, and ray-query masking
// - exact per-instance mask queries
// - host-side mask updates followed by a TLAS update
// - device-side transform/mask mutations followed by TLAS update and rebuild
// - opaque callback suppression and host/device opacity mutations

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct ExpectedHit {
    uint instance;
    uint hit_type;
};

void check_results(luisa::span<const uint4> results,
                   luisa::span<const ExpectedHit> expected,
                   luisa::string_view phase) {
    expect(results.size() == expected.size());
    for (auto i = 0u; i < results.size(); i++) {
        auto actual = results[i];
        auto exp = expected[i];
        auto should_hit = exp.hit_type == static_cast<uint>(HitType::Surface);
        expect(actual.x == exp.instance)
            << luisa::format("{} closest-hit instance mismatch at case {}: got {}, expected {}",
                             phase, i, actual.x, exp.instance);
        expect(actual.y == static_cast<uint>(should_hit))
            << luisa::format("{} any-hit mismatch at case {}: got {}, expected {}",
                             phase, i, actual.y, should_hit);
        expect(actual.w == exp.hit_type)
            << luisa::format("{} ray-query hit type mismatch at case {}: got {}, expected {}",
                             phase, i, actual.w, exp.hit_type);
        if (should_hit) {
            expect(actual.z == exp.instance)
                << luisa::format("{} ray-query instance mismatch at case {}: got {}, expected {}",
                                 phase, i, actual.z, exp.instance);
        }
    }
}

}// namespace

void test_accel_visibility(Device &device) {
    auto stream = device.create_stream();

    std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    std::array triangles{Triangle{0u, 1u, 2u}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel({.allow_update = true});
    accel.emplace_back(mesh, make_float4x4(1.0f), 0x1u, false);
    accel.emplace_back(mesh, translation(make_float3(0.0f, 0.0f, -1.0f)), 0x2u, false);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    Kernel1D trace = [](BufferUInt masks, BufferUInt4 results,
                        BufferUInt observed_masks, AccelVar accel) noexcept {
        auto index = dispatch_id().x;
        auto visibility_mask = masks.read(index);
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f));
        AccelTraceOptions options{.visibility_mask = visibility_mask};

        auto closest = accel.intersect(ray, options);
        auto any = accel.intersect_any(ray, options);
        auto committed = accel.traverse(ray, options)
                             .on_surface_candidate([](SurfaceCandidate &candidate) noexcept {
                                 candidate.commit();
                             })
                             .on_procedural_candidate([](ProceduralCandidate &) noexcept {})
                             .trace();
        results.write(index, make_uint4(
                                 closest->inst,
                                 cast<uint>(any),
                                 committed->inst,
                                 committed->hit_type));

        $if (index < 2u) {
            observed_masks.write(index, accel.instance_visibility_mask(index));
        };
    };

    auto shader = device.compile(trace);
    auto masks = device.create_buffer<uint>(8u);
    auto results = device.create_buffer<uint4>(8u);
    auto observed_masks = device.create_buffer<uint>(2u);

    auto run = [&](luisa::span<const uint> host_masks,
                   luisa::span<const ExpectedHit> expected,
                   std::array<uint, 2u> expected_instance_masks,
                   luisa::string_view phase) {
        luisa::vector<uint4> host_results(host_masks.size());
        std::array<uint, 2u> host_observed_masks{};
        stream << masks.view(0u, host_masks.size()).copy_from(host_masks)
               << shader(masks, results, observed_masks, accel).dispatch(host_masks.size())
               << results.view(0u, host_masks.size()).copy_to(luisa::span{host_results})
               << observed_masks.copy_to(luisa::span{host_observed_masks})
               << synchronize();
        check_results(luisa::span{host_results}, expected, phase);
        expect(host_observed_masks[0] == expected_instance_masks[0])
            << luisa::format("{} instance 0 visibility query mismatch", phase);
        expect(host_observed_masks[1] == expected_instance_masks[1])
            << luisa::format("{} instance 1 visibility query mismatch", phase);
    };

    static constexpr auto miss = ~0u;
    static constexpr auto surface = static_cast<uint>(HitType::Surface);
    static constexpr auto no_hit = static_cast<uint>(HitType::Miss);

    const std::array initial_masks{0x1u, 0x2u, 0x3u, 0x4u,
                                   ~0x1u, ~0x2u, 0x0u};
    const std::array initial_expected{
        ExpectedHit{0u, surface}, ExpectedHit{1u, surface},
        ExpectedHit{0u, surface}, ExpectedHit{miss, no_hit},
        ExpectedHit{1u, surface}, ExpectedHit{0u, surface},
        ExpectedHit{miss, no_hit}};
    run(initial_masks, initial_expected, {0x1u, 0x2u}, "initial build");

    accel.set_visibility_on_update(0u, 0x4u);
    accel.set_visibility_on_update(1u, 0x8u);
    stream << accel.build() << synchronize();

    const std::array updated_masks{0x1u, 0x2u, 0x4u, 0x8u,
                                   0xcu, ~0x4u, ~0x8u, 0x0u};
    const std::array updated_expected{
        ExpectedHit{miss, no_hit}, ExpectedHit{miss, no_hit},
        ExpectedHit{0u, surface}, ExpectedHit{1u, surface},
        ExpectedHit{0u, surface}, ExpectedHit{1u, surface},
        ExpectedHit{0u, surface}, ExpectedHit{miss, no_hit}};
    run(updated_masks, updated_expected, {0x4u, 0x8u}, "TLAS update");
}

void test_accel_device_mutation(Device &device) {
    auto stream = device.create_stream();

    std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    std::array triangles{Triangle{0u, 1u, 2u}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel({.allow_update = true});
    accel.emplace_back(mesh, make_float4x4(1.0f), 0x1u, false);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    Kernel1D mutate = [](AccelVar accel, Float4x4 transform,
                         UInt visibility_mask) noexcept {
        accel.set_instance_transform(0u, transform);
        accel.set_instance_visibility(0u, visibility_mask);
    };
    Kernel1D trace = [](BufferFloat3 origins, BufferUInt masks,
                        BufferUInt4 results, BufferUInt2 observed,
                        AccelVar accel) noexcept {
        auto index = dispatch_id().x;
        auto ray = make_ray(origins.read(index), make_float3(0.0f, 0.0f, -1.0f));
        AccelTraceOptions options{.visibility_mask = masks.read(index)};
        auto closest = accel.intersect(ray, options);
        auto any = accel.intersect_any(ray, options);
        auto committed = accel.traverse(ray, options)
                             .on_surface_candidate([](SurfaceCandidate &candidate) noexcept {
                                 candidate.commit();
                             })
                             .on_procedural_candidate([](ProceduralCandidate &) noexcept {})
                             .trace();
        results.write(index, make_uint4(
                                 closest->inst,
                                 cast<uint>(any),
                                 committed->inst,
                                 committed->hit_type));
        $if (index == 0u) {
            observed.write(0u, make_uint2(
                                   accel.instance_visibility_mask(0u),
                                   accel.instance_user_id(0u)));
        };
    };

    auto mutate_shader = device.compile(mutate);
    auto trace_shader = device.compile(trace);
    auto origins = device.create_buffer<float3>(3u);
    auto masks = device.create_buffer<uint>(3u);
    auto results = device.create_buffer<uint4>(3u);
    auto observed = device.create_buffer<uint2>(1u);

    auto run = [&](const std::array<float3, 3u> &host_origins,
                   const std::array<uint, 3u> &host_masks,
                   const std::array<ExpectedHit, 3u> &expected,
                   uint expected_mask, uint expected_user_id,
                   luisa::string_view phase) {
        std::array<uint4, 3u> host_results{};
        std::array<uint2, 1u> host_observed{};
        stream << origins.copy_from(luisa::span{host_origins})
               << masks.copy_from(luisa::span{host_masks})
               << trace_shader(origins, masks, results, observed, accel).dispatch(3u)
               << results.copy_to(luisa::span{host_results})
               << observed.copy_to(luisa::span{host_observed})
               << synchronize();
        check_results(luisa::span{host_results}, luisa::span{expected}, phase);
        expect(host_observed[0].x == expected_mask)
            << luisa::format("{} device visibility query mismatch", phase);
        expect(host_observed[0].y == expected_user_id)
            << luisa::format("{} unrelated host user-id update was lost", phase);
    };

    static constexpr auto miss = ~0u;
    static constexpr auto surface = static_cast<uint>(HitType::Surface);
    static constexpr auto no_hit = static_cast<uint>(HitType::Miss);

    // The unrelated host user-id update is deliberately committed in the same
    // build command. It must not overwrite the GPU-authored transform or mask.
    accel.set_instance_user_id_on_update(0u, 17u);
    stream << mutate_shader(
                  accel, translation(make_float3(2.0f, 0.0f, 0.0f)), 0x4u)
                  .dispatch(1u)
           << accel.build(Accel::BuildRequest::PREFER_UPDATE);
    const std::array update_origins{
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(2.0f, 0.0f, 1.0f),
        make_float3(2.0f, 0.0f, 1.0f)};
    const std::array update_masks{0x4u, 0x1u, 0x4u};
    const std::array update_expected{
        ExpectedHit{miss, no_hit},
        ExpectedHit{miss, no_hit},
        ExpectedHit{0u, surface}};
    run(update_origins, update_masks, update_expected, 0x4u, 17u,
        "device mutation TLAS update");

    stream << mutate_shader(
                  accel, translation(make_float3(-2.0f, 0.0f, 0.0f)), 0x8u)
                  .dispatch(1u)
           << accel.build(Accel::BuildRequest::FORCE_BUILD);
    const std::array rebuild_origins{
        make_float3(2.0f, 0.0f, 1.0f),
        make_float3(-2.0f, 0.0f, 1.0f),
        make_float3(-2.0f, 0.0f, 1.0f)};
    const std::array rebuild_masks{0x8u, 0x4u, 0x8u};
    const std::array rebuild_expected{
        ExpectedHit{miss, no_hit},
        ExpectedHit{miss, no_hit},
        ExpectedHit{0u, surface}};
    run(rebuild_origins, rebuild_masks, rebuild_expected, 0x8u, 17u,
        "device mutation TLAS rebuild");
}

void test_accel_opacity(Device &device) {
    auto stream = device.create_stream();

    std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    std::array triangles{Triangle{0u, 1u, 2u}};

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto accel = device.create_accel({.allow_update = true});
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, true);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << accel.build()
           << synchronize();

    // Store one result for closest ray query and one for ANY ray query:
    // {hit type, instance, primitive, surface callback count}.
    Kernel1D trace = [](BufferUInt4 results, AccelVar accel) noexcept {
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f));

        UInt closest_callback_count = 0u;
        auto closest = accel.traverse(ray, {})
                           .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                               closest_callback_count += 1u;
                               candidate.commit();
                           })
                           .on_procedural_candidate([](ProceduralCandidate &) noexcept {})
                           .trace();
        results.write(0u, make_uint4(
                              closest->hit_type, closest->inst,
                              closest->prim, closest_callback_count));

        UInt any_callback_count = 0u;
        auto any = accel.traverse_any(ray, {})
                       .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                           any_callback_count += 1u;
                           candidate.commit();
                       })
                       .on_procedural_candidate([](ProceduralCandidate &) noexcept {})
                       .trace();
        results.write(1u, make_uint4(
                              any->hit_type, any->inst,
                              any->prim, any_callback_count));
    };

    Kernel1D mutate = [](AccelVar accel, Bool opaque) noexcept {
        accel.set_instance_opaque(0u, opaque);
    };

    auto trace_shader = device.compile(trace);
    auto mutate_shader = device.compile(mutate);
    auto results = device.create_buffer<uint4>(2u);

    auto run = [&](uint expected_callback_count,
                   luisa::string_view phase) {
        std::array<uint4, 2u> host_results{};
        stream << trace_shader(results, accel).dispatch(1u)
               << results.copy_to(luisa::span{host_results})
               << synchronize();
        for (auto query = 0u; query < 2u; query++) {
            auto result = host_results[query];
            auto query_name = query == 0u ? "closest" : "any";
            expect(result.x == static_cast<uint>(HitType::Surface))
                << luisa::format("{} {} query did not auto/explicitly commit the surface",
                                 phase, query_name);
            expect(result.y == 0u)
                << luisa::format("{} {} query committed the wrong instance: {}",
                                 phase, query_name, result.y);
            expect(result.z == 0u)
                << luisa::format("{} {} query committed the wrong primitive: {}",
                                 phase, query_name, result.z);
            expect(result.w == expected_callback_count)
                << luisa::format("{} {} query callback count mismatch: got {}, expected {}",
                                 phase, query_name, result.w, expected_callback_count);
        }
    };

    // Opaque hits must commit in traversal and skip user filtering.
    run(0u, "initial host opaque");

    accel.set_opaque_on_update(0u, false);
    stream << accel.build(Accel::BuildRequest::PREFER_UPDATE);
    run(1u, "host non-opaque update");

    accel.set_opaque_on_update(0u, true);
    stream << accel.build(Accel::BuildRequest::PREFER_UPDATE);
    run(0u, "host opaque update");

    // Device-side mutation must use the same opacity bit as host updates.
    stream << mutate_shader(accel, false).dispatch(1u)
           << accel.build(Accel::BuildRequest::PREFER_UPDATE);
    run(1u, "device non-opaque update");

    stream << mutate_shader(accel, true).dispatch(1u)
           << accel.build(Accel::BuildRequest::PREFER_UPDATE);
    run(0u, "device opaque update");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_accel_visibility(dc->device);
    test_accel_device_mutation(dc->device);
    test_accel_opacity(dc->device);
}
