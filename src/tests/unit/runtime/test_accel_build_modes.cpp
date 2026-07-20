// Test for acceleration-structure build modes and mutable geometry.
// This test covers:
// - FAST_TRACE and FAST_BUILD usage hints
// - update-enabled compacted and non-compacted BLAS/TLAS builds
// - triangle and procedural BLAS updates and forced rebuilds
// - exact ray-query hit type, instance, primitive, and callback results

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>
#include <cstddef>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct ExpectedResult {
    uint32_t hit_type;
    uint32_t instance;
    uint32_t primitive;
    uint32_t callback_kind;
};

[[nodiscard]] auto make_triangle_vertices(float x) noexcept {
    return std::array{
        make_float3(x - 0.75f, -2.75f, 0.0f),
        make_float3(x + 0.75f, -2.75f, 0.0f),
        make_float3(x, -1.25f, 0.0f)};
}

[[nodiscard]] auto make_aabb(float x) noexcept {
    return AABB{
        .packed_min = {x - 0.5f, 1.5f, -0.1f},
        .packed_max = {x + 0.5f, 2.5f, 0.1f}};
}

[[nodiscard]] auto make_expected(std::size_t mesh_ray, std::size_t procedural_ray) noexcept {
    static constexpr auto miss = static_cast<uint32_t>(HitType::Miss);
    static constexpr auto surface = static_cast<uint32_t>(HitType::Surface);
    static constexpr auto procedural = static_cast<uint32_t>(HitType::Procedural);
    static constexpr ExpectedResult no_hit{miss, ~0u, ~0u, 0u};
    std::array expected{no_hit, no_hit, no_hit, no_hit, no_hit, no_hit};
    expected[mesh_ray] = ExpectedResult{surface, 0u, 0u, 1u};
    expected[procedural_ray] = ExpectedResult{procedural, 1u, 0u, 2u};
    return expected;
}

void check_results(luisa::span<const uint4> actual,
                   luisa::span<const ExpectedResult> expected,
                   luisa::string_view phase) {
    expect(actual.size() == expected.size());
    for (auto i = 0u; i < actual.size(); i++) {
        auto result = actual[i];
        auto reference = expected[i];
        expect(result.x == reference.hit_type)
            << luisa::format("{} hit type mismatch at ray {}: got {}, expected {}",
                             phase, i, result.x, reference.hit_type);
        expect(result.y == reference.instance)
            << luisa::format("{} instance mismatch at ray {}: got {}, expected {}",
                             phase, i, result.y, reference.instance);
        expect(result.z == reference.primitive)
            << luisa::format("{} primitive mismatch at ray {}: got {}, expected {}",
                             phase, i, result.z, reference.primitive);
        expect(result.w == reference.callback_kind)
            << luisa::format("{} callback kind mismatch at ray {}: got {}, expected {}",
                             phase, i, result.w, reference.callback_kind);
    }
}

}// namespace

void test_accel_build_modes(Device &device) {
    static constexpr std::size_t ray_count = 6u;
    auto stream = device.create_stream();

    // The first three rays target the triangle row; the last three target the
    // procedural row. Within each row they target x = -2, 0, and 2.
    const std::array origins{
        make_float3(-2.0f, -2.0f, 1.0f),
        make_float3(0.0f, -2.0f, 1.0f),
        make_float3(2.0f, -2.0f, 1.0f),
        make_float3(-2.0f, 2.0f, 1.0f),
        make_float3(0.0f, 2.0f, 1.0f),
        make_float3(2.0f, 2.0f, 1.0f)};
    auto origin_buffer = device.create_buffer<float3>(origins.size());
    auto result_buffer = device.create_buffer<uint4>(origins.size());
    stream << origin_buffer.copy_from(luisa::span{origins}) << synchronize();

    Kernel1D trace = [](BufferFloat3 ray_origins,
                        BufferUInt4 results,
                        AccelVar accel) noexcept {
        auto index = dispatch_id().x;
        auto ray = make_ray(ray_origins.read(index),
                            make_float3(0.0f, 0.0f, -1.0f));
        UInt callback_kind = 0u;
        auto committed = accel.traverse(ray, {})
                             .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                                 callback_kind = 1u;
                                 candidate.commit();
                             })
                             .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                                 callback_kind = 2u;
                                 candidate.commit(1.0f);
                             })
                             .trace();
        auto is_miss = committed->hit_type == static_cast<uint32_t>(HitType::Miss);
        // Instance/primitive values are unspecified for a miss on some
        // backends, so normalize them before exact host-side comparison.
        results.write(index, make_uint4(
                                 committed->hit_type,
                                 ite(is_miss, ~0u, committed->inst),
                                 ite(is_miss, ~0u, committed->prim),
                                 callback_kind));
    };
    auto trace_shader = device.compile(trace);

    constexpr std::array hints{
        AccelUsageHint::FAST_TRACE,
        AccelUsageHint::FAST_BUILD};
    constexpr std::array compaction_modes{false, true};
    const std::array triangles{Triangle{0u, 1u, 2u}};

    for (auto hint : hints) {
        for (auto allow_compaction : compaction_modes) {
            auto hint_name = hint == AccelUsageHint::FAST_TRACE ?
                                 luisa::string_view{"FAST_TRACE"} :
                                 luisa::string_view{"FAST_BUILD"};
            auto mode_name = luisa::format("{} compact={}", hint_name, allow_compaction);
            LUISA_INFO("Testing acceleration build mode: {}", mode_name);

            AccelOption option{
                .hint = hint,
                .allow_compaction = allow_compaction,
                .allow_update = true};
            auto vertices = make_triangle_vertices(-2.0f);
            std::array aabbs{make_aabb(-2.0f)};
            auto vertex_buffer = device.create_buffer<float3>(vertices.size());
            auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
            auto aabb_buffer = device.create_buffer<AABB>(aabbs.size());
            auto mesh = device.create_mesh(vertex_buffer, triangle_buffer, option);
            auto procedural = device.create_procedural_primitive(aabb_buffer, option);
            auto accel = device.create_accel(option);
            // Force the surface callback so callback kind is also checked.
            accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
            accel.emplace_back(procedural);

            auto run = [&](const std::array<ExpectedResult, ray_count> &expected,
                           luisa::string_view phase) {
                std::array<uint4, ray_count> host_results{};
                stream << trace_shader(origin_buffer, result_buffer, accel).dispatch(origins.size())
                       << result_buffer.copy_to(luisa::span{host_results})
                       << synchronize();
                auto label = luisa::format("{} {}", mode_name, phase);
                check_results(luisa::span{host_results}, luisa::span{expected}, label);
            };

            stream << vertex_buffer.copy_from(luisa::span{vertices})
                   << triangle_buffer.copy_from(luisa::span{triangles})
                   << aabb_buffer.copy_from(luisa::span{aabbs})
                   << mesh.build(AccelBuildRequest::FORCE_BUILD)
                   << procedural.build(AccelBuildRequest::FORCE_BUILD)
                   << accel.build(AccelBuildRequest::FORCE_BUILD);
            run(make_expected(0u, 3u), "initial build");

            vertices = make_triangle_vertices(0.0f);
            aabbs[0] = make_aabb(0.0f);
            stream << vertex_buffer.copy_from(luisa::span{vertices})
                   << aabb_buffer.copy_from(luisa::span{aabbs})
                   << mesh.build(AccelBuildRequest::PREFER_UPDATE)
                   << procedural.build(AccelBuildRequest::PREFER_UPDATE)
                   << accel.build(AccelBuildRequest::PREFER_UPDATE);
            run(make_expected(1u, 4u), "prefer update");

            vertices = make_triangle_vertices(2.0f);
            aabbs[0] = make_aabb(2.0f);
            stream << vertex_buffer.copy_from(luisa::span{vertices})
                   << aabb_buffer.copy_from(luisa::span{aabbs})
                   << mesh.build(AccelBuildRequest::FORCE_BUILD)
                   << procedural.build(AccelBuildRequest::FORCE_BUILD)
                   << accel.build(AccelBuildRequest::FORCE_BUILD);
            run(make_expected(2u, 5u), "force rebuild");
        }
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_accel_build_modes(dc->device);
}
