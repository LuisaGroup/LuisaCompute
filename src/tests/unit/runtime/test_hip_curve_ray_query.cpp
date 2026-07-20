// Exact HIP XIR ray-query coverage for curve surface candidates.
// This test verifies opaque auto-commit, non-opaque accept/reject callbacks,
// ALL/ANY query modes, curve classification, hit IDs/parameters, and the
// committed world-ray distance exposed from a callback.

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

struct ExpectedCase {
    uint hit_type;
    uint committed_inst;
    uint committed_prim;
    uint callback_count;
    bool has_committed_curve;
    float committed_t;
    bool has_callback;
    uint callback_inst;
    uint callback_prim;
    float candidate_t;
    float candidate_u;
    float candidate_tmax_after;
};

void expect_near(luisa::string_view query, size_t index,
                 luisa::string_view field, float actual, float expected) {
    constexpr auto tolerance = 2.0e-4f;
    expect(std::abs(actual - expected) < tolerance)
        << luisa::format("{} case {} {}: got {}, expected {}",
                         query, index, field, actual, expected);
}

void check_results(
    luisa::string_view query,
    luisa::span<const uint4> summaries,
    luisa::span<const float4> committed_details,
    luisa::span<const uint4> callback_metadata,
    luisa::span<const float4> candidate_details,
    luisa::span<const float> candidate_tmax_after,
    luisa::span<const ExpectedCase> expected) {
    expect(summaries.size() == expected.size());
    expect(committed_details.size() == expected.size());
    expect(callback_metadata.size() == expected.size());
    expect(candidate_details.size() == expected.size());
    expect(candidate_tmax_after.size() == expected.size());

    for (auto i = 0u; i < expected.size(); i++) {
        auto summary = summaries[i];
        auto committed = committed_details[i];
        auto callback = callback_metadata[i];
        auto candidate = candidate_details[i];
        auto exp = expected[i];

        expect(summary.x == exp.hit_type)
            << luisa::format("{} case {} hit type: got {}, expected {}",
                             query, i, summary.x, exp.hit_type);
        expect(summary.w == exp.callback_count)
            << luisa::format("{} case {} callback count: got {}, expected {}",
                             query, i, summary.w, exp.callback_count);

        if (exp.has_committed_curve) {
            expect(summary.y == exp.committed_inst)
                << luisa::format("{} case {} committed instance: got {}, expected {}",
                                 query, i, summary.y, exp.committed_inst);
            expect(summary.z == exp.committed_prim)
                << luisa::format("{} case {} committed primitive: got {}, expected {}",
                                 query, i, summary.z, exp.committed_prim);
            expect_near(query, i, "committed distance", committed.x, exp.committed_t);
            expect_near(query, i, "committed curve parameter", committed.y, 0.5f);
            expect_near(query, i, "committed curve marker", committed.z, -1.0f);
            expect_near(query, i, "committed curve classification", committed.w, 1.0f);
        } else {
            expect_near(query, i, "miss curve classification", committed.w, 0.0f);
        }

        if (exp.has_callback) {
            expect(callback.x == 1u)
                << luisa::format("{} case {} callback did not classify the candidate as a curve",
                                 query, i);
            expect(callback.y == exp.callback_inst)
                << luisa::format("{} case {} candidate instance: got {}, expected {}",
                                 query, i, callback.y, exp.callback_inst);
            expect(callback.z == exp.callback_prim)
                << luisa::format("{} case {} candidate primitive: got {}, expected {}",
                                 query, i, callback.z, exp.callback_prim);
            expect(callback.w == exp.callback_count);
            expect_near(query, i, "candidate distance", candidate.x, exp.candidate_t);
            expect_near(query, i, "candidate curve parameter", candidate.y, exp.candidate_u);
            expect_near(query, i, "candidate curve marker", candidate.z, -1.0f);
            expect_near(query, i, "candidate pre-commit ray tmax", candidate.w, 1.0f);
            expect_near(query, i, "candidate post-commit ray tmax",
                        candidate_tmax_after[i], exp.candidate_tmax_after);
        } else {
            expect(callback.x == 0u);
            expect(callback.y == ~0u);
            expect(callback.z == ~0u);
            expect(callback.w == 0u);
            expect_near(query, i, "untouched candidate distance", candidate.x, -1.0f);
            expect_near(query, i, "untouched candidate post-commit tmax",
                        candidate_tmax_after[i], -1.0f);
        }
    }
}

void test_hip_curve_ray_query(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO("Skipping HIP-specific curve ray-query test on backend '{}'.",
                   device.backend_name());
        return;
    }

    constexpr auto curve_basis = CurveBasis::PIECEWISE_LINEAR;
    const std::array control_points{
        make_float4(-0.5f, 0.0f, 0.0f, 0.1f),
        make_float4(0.5f, 0.0f, 0.0f, 0.1f),
        make_float4(-0.5f, 1.0f, 0.1f, 0.1f),
        make_float4(0.5f, 1.0f, 0.1f, 0.1f)};
    const std::array segments{0u, 2u};

    auto stream = device.create_stream();
    auto control_point_buffer = device.create_buffer<float4>(control_points.size());
    auto segment_buffer = device.create_buffer<uint>(segments.size());
    auto curve = device.create_curve(
        curve_basis, control_point_buffer, segment_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(curve, translation(make_float3(-2.0f, 0.0f, 0.0f)),
                       0xffu, true);
    accel.emplace_back(curve, make_float4x4(1.0f), 0xffu, false);

    stream << control_point_buffer.copy_from(luisa::span{control_points})
           << segment_buffer.copy_from(luisa::span{segments})
           << curve.build()
           << accel.build()
           << synchronize();

    auto write_results = [](UInt index, const Var<CommittedHit> &committed,
                            UInt callback_count, UInt callback_curve,
                            UInt callback_inst, UInt callback_prim,
                            Float candidate_t, Float candidate_u,
                            Float candidate_v, Float candidate_tmax_before,
                            Float candidate_tmax_after,
                            BufferUInt4 &summaries,
                            BufferFloat4 &committed_details,
                            BufferUInt4 &callback_metadata,
                            BufferFloat4 &candidate_details,
                            BufferFloat &output_candidate_tmax_after) noexcept {
        summaries.write(index, make_uint4(
                                   committed->hit_type,
                                   committed->inst,
                                   committed->prim,
                                   callback_count));
        committed_details.write(index, make_float4(
                                           committed->distance(),
                                           committed->curve_parameter(),
                                           committed->bary.y,
                                           cast<float>(committed->is_curve())));
        callback_metadata.write(index, make_uint4(
                                           callback_curve,
                                           callback_inst,
                                           callback_prim,
                                           callback_count));
        candidate_details.write(index, make_float4(
                                           candidate_t,
                                           candidate_u,
                                           candidate_v,
                                           candidate_tmax_before));
        output_candidate_tmax_after.write(index, candidate_tmax_after);
    };

    Kernel1D trace_all = [write_results](
                             AccelVar accel,
                             BufferUInt4 summaries,
                             BufferFloat4 committed_details,
                             BufferUInt4 callback_metadata,
                             BufferFloat4 candidate_details,
                             BufferFloat candidate_tmax_after) noexcept {
        auto index = dispatch_id().x;
        auto origin_x = ite(index == 0u, -2.0f, 0.0f);
        auto origin_y = ite(index == 2u, 1.0f, 0.0f);
        auto accept = index == 1u;
        auto ray = make_ray(make_float3(origin_x, origin_y, 1.0f),
                            make_float3(0.0f, 0.0f, -2.0f), 0.0f, 1.0f);

        UInt callback_count = 0u;
        UInt callback_curve = 0u;
        UInt callback_inst = ~0u;
        UInt callback_prim = ~0u;
        Float candidate_t = -1.0f;
        Float candidate_u = -1.0f;
        Float candidate_v = -1.0f;
        Float candidate_tmax_before = -1.0f;
        Float candidate_tmax_committed = -1.0f;

        auto committed = accel.traverse(
                                  ray, {.curve_bases = {curve_basis}})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     callback_count += 1u;
                                     callback_curve = cast<uint>(hit->is_curve());
                                     callback_inst = hit->inst;
                                     callback_prim = hit->prim;
                                     candidate_t = hit->distance();
                                     candidate_u = hit->curve_parameter();
                                     candidate_v = hit->bary.y;
                                     candidate_tmax_before = candidate.ray()->t_max();
                                     $if (accept) {
                                         candidate.commit();
                                         candidate_tmax_committed = candidate.ray()->t_max();
                                     }
                                     $else {
                                         candidate_tmax_committed = candidate.ray()->t_max();
                                     };
                                 })
                             .trace();
        write_results(index, committed,
                      callback_count, callback_curve,
                      callback_inst, callback_prim,
                      candidate_t, candidate_u, candidate_v,
                      candidate_tmax_before, candidate_tmax_committed,
                      summaries, committed_details, callback_metadata,
                      candidate_details, candidate_tmax_after);
    };

    Kernel1D trace_any = [write_results](
                             AccelVar accel,
                             BufferUInt4 summaries,
                             BufferFloat4 committed_details,
                             BufferUInt4 callback_metadata,
                             BufferFloat4 candidate_details,
                             BufferFloat candidate_tmax_after) noexcept {
        auto index = dispatch_id().x;
        auto origin_x = ite(index == 0u, -2.0f, 0.0f);
        auto origin_y = ite(index == 2u, 0.0f, 1.0f);
        auto accept = index == 1u;
        auto ray = make_ray(make_float3(origin_x, origin_y, 1.0f),
                            make_float3(0.0f, 0.0f, -2.0f), 0.0f, 1.0f);

        UInt callback_count = 0u;
        UInt callback_curve = 0u;
        UInt callback_inst = ~0u;
        UInt callback_prim = ~0u;
        Float candidate_t = -1.0f;
        Float candidate_u = -1.0f;
        Float candidate_v = -1.0f;
        Float candidate_tmax_before = -1.0f;
        Float candidate_tmax_committed = -1.0f;

        auto committed = accel.traverse_any(
                                  ray, {.curve_bases = {curve_basis}})
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     auto hit = candidate.hit();
                                     callback_count += 1u;
                                     callback_curve = cast<uint>(hit->is_curve());
                                     callback_inst = hit->inst;
                                     callback_prim = hit->prim;
                                     candidate_t = hit->distance();
                                     candidate_u = hit->curve_parameter();
                                     candidate_v = hit->bary.y;
                                     candidate_tmax_before = candidate.ray()->t_max();
                                     $if (accept) {
                                         candidate.commit();
                                         candidate_tmax_committed = candidate.ray()->t_max();
                                     }
                                     $else {
                                         candidate_tmax_committed = candidate.ray()->t_max();
                                     };
                                 })
                             .trace();
        write_results(index, committed,
                      callback_count, callback_curve,
                      callback_inst, callback_prim,
                      candidate_t, candidate_u, candidate_v,
                      candidate_tmax_before, candidate_tmax_committed,
                      summaries, committed_details, callback_metadata,
                      candidate_details, candidate_tmax_after);
    };

    auto all_shader = device.compile(trace_all);
    auto any_shader = device.compile(trace_any);
    auto summaries = device.create_buffer<uint4>(3u);
    auto committed_details = device.create_buffer<float4>(3u);
    auto callback_metadata = device.create_buffer<uint4>(3u);
    auto candidate_details = device.create_buffer<float4>(3u);
    auto candidate_tmax_after = device.create_buffer<float>(3u);

    std::array<uint4, 3u> host_summaries{};
    std::array<float4, 3u> host_committed_details{};
    std::array<uint4, 3u> host_callback_metadata{};
    std::array<float4, 3u> host_candidate_details{};
    std::array<float, 3u> host_candidate_tmax_after{};

    auto dispatch_and_download = [&](auto &shader) {
        stream << shader(accel, summaries, committed_details,
                         callback_metadata, candidate_details,
                         candidate_tmax_after)
                      .dispatch(3u)
               << summaries.copy_to(luisa::span{host_summaries})
               << committed_details.copy_to(luisa::span{host_committed_details})
               << callback_metadata.copy_to(luisa::span{host_callback_metadata})
               << candidate_details.copy_to(luisa::span{host_candidate_details})
               << candidate_tmax_after.copy_to(luisa::span{host_candidate_tmax_after})
               << synchronize();
    };

    constexpr auto surface = static_cast<uint>(HitType::Surface);
    constexpr auto miss = static_cast<uint>(HitType::Miss);
    const std::array expected_all{
        ExpectedCase{surface, 0u, 0u, 0u, true, 0.45f,
                     false, ~0u, ~0u, -1.0f, -1.0f, -1.0f},
        ExpectedCase{surface, 1u, 0u, 1u, true, 0.45f,
                     true, 1u, 0u, 0.45f, 0.5f, 0.45f},
        ExpectedCase{miss, ~0u, ~0u, 1u, false, 1.0f,
                     true, 1u, 1u, 0.40f, 0.5f, 1.0f}};
    dispatch_and_download(all_shader);
    check_results("ALL", host_summaries, host_committed_details,
                  host_callback_metadata, host_candidate_details,
                  host_candidate_tmax_after, expected_all);

    const std::array expected_any{
        ExpectedCase{surface, 0u, 1u, 0u, true, 0.40f,
                     false, ~0u, ~0u, -1.0f, -1.0f, -1.0f},
        ExpectedCase{surface, 1u, 1u, 1u, true, 0.40f,
                     true, 1u, 1u, 0.40f, 0.5f, 0.40f},
        ExpectedCase{miss, ~0u, ~0u, 1u, false, 1.0f,
                     true, 1u, 0u, 0.45f, 0.5f, 1.0f}};
    dispatch_and_download(any_shader);
    check_results("ANY", host_summaries, host_committed_details,
                  host_callback_metadata, host_candidate_details,
                  host_candidate_tmax_after, expected_any);
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP XIR curve ray queries preserve exact surface semantics"_test = [&] {
        test_hip_curve_ray_query(dc->device);
    };
}
