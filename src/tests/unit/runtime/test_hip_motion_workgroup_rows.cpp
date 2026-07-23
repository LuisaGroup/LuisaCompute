// Test for HIP direct motion tracing across two-dimensional workgroups.
// This test covers:
// - closest-hit and any-hit motion traversal with an explicit 16x16 block
// - repeated launches over every local row in a 32x32 dispatch, with
//   lane-dependent barycentrics
// - identity-key and translating SRT motion at two interior times
// - outer instance, primitive, user ID, distance, and barycentric results

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

constexpr auto dispatch_width = 32u;
constexpr auto dispatch_height = 32u;
constexpr auto block_width = 16u;
constexpr auto block_height = 16u;
constexpr auto element_count = dispatch_width * dispatch_height;
constexpr auto repeat_count = 64u;
constexpr auto result_count = element_count * repeat_count;
constexpr auto expected_motion_instance_index = 1u;
constexpr auto expected_primitive_index = 0u;
constexpr auto identity_user_id = 0x00112233u;
constexpr auto translating_user_id = 0x00445566u;
constexpr auto expected_distance = 1.0f;
constexpr auto float_tolerance = 5.0e-4f;

struct ProbeSet {
    luisa::vector<float4> probes;
    luisa::vector<float2> barycentrics;
};

[[nodiscard]] ProbeSet make_probes(bool translating) {
    ProbeSet result;
    result.probes.resize(element_count);
    result.barycentrics.resize(element_count);
    for (auto y = 0u; y < dispatch_height; y++) {
        for (auto x = 0u; x < dispatch_width; x++) {
            auto local_x = x % block_width;
            auto local_y = y % block_height;
            auto u = 0.20f + 0.01f * static_cast<float>(local_x);
            auto v = 0.30f + 0.005f * static_cast<float>(local_y);
            // p = (1 - u - v) * v0 + u * v1 + v * v2 for
            // v0=(-0.5,-0.5), v1=(0.5,-0.5), v2=(0,0.5).
            // The remaining barycentric weight is at least 0.275, keeping all
            // rays well away from triangle edges.
            auto point = make_float2(-0.5f + u + 0.5f * v,
                                     -0.5f + v);
            auto time = y < block_height ? 0.25f : 0.75f;
            if (translating) {
                point += make_float2(-2.0f + 4.0f * time, 3.0f);
            }
            auto index = x + y * dispatch_width;
            result.probes[index] = make_float4(point.x, point.y, time, 0.0f);
            result.barycentrics[index] = make_float2(u, v);
        }
    }
    return result;
}

void check_results(luisa::string_view phase,
                   luisa::span<const uint4> summaries,
                   luisa::span<const float4> details,
                   luisa::span<const float2> expected_barycentrics,
                   uint32_t expected_user_id) {
    expect(summaries.size() == result_count);
    expect(details.size() == result_count);
    expect(expected_barycentrics.size() == element_count);

    std::array<uint32_t, block_height> valid_per_local_row{};
    auto mismatch_count = 0u;
    constexpr auto diagnostic_limit = 32u;
    for (auto repeat = 0u; repeat < repeat_count; repeat++) {
        for (auto y = 0u; y < dispatch_height; y++) {
            for (auto x = 0u; x < dispatch_width; x++) {
                auto pixel_index = x + y * dispatch_width;
                auto result_index = pixel_index + repeat * element_count;
                auto summary = summaries[result_index];
                auto detail = details[result_index];
                auto expected_barycentric = expected_barycentrics[pixel_index];
                auto valid = summary.x == expected_motion_instance_index &&
                             summary.y == expected_primitive_index &&
                             summary.z == expected_user_id &&
                             summary.w == 3u &&
                             std::abs(detail.x - expected_distance) < float_tolerance &&
                             std::abs(detail.y - expected_barycentric.x) < float_tolerance &&
                             std::abs(detail.z - expected_barycentric.y) < float_tolerance &&
                             std::abs(detail.w - 1.0f) < float_tolerance;
                if (valid) {
                    valid_per_local_row[y % block_height]++;
                } else {
                    if (mismatch_count < diagnostic_limit) {
                        LUISA_WARNING(
                            "{} repeat {} at global ({}, {}), local ({}, {}): "
                            "summary=({}, {}, 0x{:08x}, {}), "
                            "detail=({}, {}, {}, {}), expected bary=({}, {})",
                            phase, repeat, x, y, x % block_width, y % block_height,
                            summary.x, summary.y, summary.z, summary.w,
                            detail.x, detail.y, detail.z, detail.w,
                            expected_barycentric.x, expected_barycentric.y);
                    }
                    mismatch_count++;
                }
            }
        }
    }

    constexpr auto expected_per_local_row = result_count / block_height;
    for (auto local_y = 0u; local_y < block_height; local_y++) {
        expect(valid_per_local_row[local_y] == expected_per_local_row)
            << luisa::format(
                   "{} local row {}: {}/{} complete direct-motion results were valid",
                   phase, local_y, valid_per_local_row[local_y],
                   expected_per_local_row);
    }
    expect(mismatch_count == 0u)
        << luisa::format("{} had {} invalid direct-motion results",
                         phase, mismatch_count);
}

void test_hip_motion_workgroup_rows(Device &device) {
    if (device.backend_name() != "hip") {
        LUISA_INFO(
            "Skipping HIP-specific motion workgroup-row test on backend '{}'.",
            device.backend_name());
        return;
    }

    log_level_verbose();

    const std::array vertices{
        make_float3(-0.5f, -0.5f, 0.0f),
        make_float3(0.5f, -0.5f, 0.0f),
        make_float3(0.0f, 0.5f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};

    auto stream = device.create_stream();
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);

    AccelMotionOption motion_option{};
    motion_option.keyframe_count = 2u;
    motion_option.time_start = 0.0f;
    motion_option.time_end = 1.0f;
    motion_option.mode = AccelMotionMode::SRT;

    auto identity_motion = device.create_motion_instance(mesh, motion_option);
    const std::array identity_keyframes{
        MotionInstanceTransformSRT{},
        MotionInstanceTransformSRT{}};
    identity_motion.set_keyframes(luisa::span{identity_keyframes});

    auto translating_motion = device.create_motion_instance(mesh, motion_option);
    std::array translating_keyframes{
        MotionInstanceTransformSRT{},
        MotionInstanceTransformSRT{}};
    translating_keyframes[0].translation[0] = -2.0f;
    translating_keyframes[0].translation[1] = 3.0f;
    translating_keyframes[1].translation[0] = 2.0f;
    translating_keyframes[1].translation[1] = 3.0f;
    translating_motion.set_keyframes(luisa::span{translating_keyframes});

    auto identity_accel = device.create_accel();
    auto translating_accel = device.create_accel();
    // A far-away decoy makes the expected outer instance index different
    // from the nested motion scene's child index.
    identity_accel.emplace_back(mesh, translation(-100.0f, 0.0f, 0.0f),
                                0xffu, true, 7u);
    identity_accel.emplace_back(identity_motion, make_float4x4(1.0f),
                                0xffu, true, identity_user_id);
    translating_accel.emplace_back(mesh, translation(-100.0f, 0.0f, 0.0f),
                                   0xffu, true, 11u);
    translating_accel.emplace_back(translating_motion, make_float4x4(1.0f),
                                   0xffu, true, translating_user_id);

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build()
           << identity_motion.build()
           << translating_motion.build()
           << identity_accel.build()
           << translating_accel.build()
           << synchronize();

    Kernel2D trace = [](AccelVar accel, BufferFloat4 probes,
                        BufferUInt4 summaries,
                        BufferFloat4 details,
                        UInt repeat_index) noexcept {
        set_block_size(block_width, block_height, 1u);
        auto coord = dispatch_id().xy();
        auto pixel_index = coord.x + coord.y * dispatch_width;
        auto result_index = pixel_index + repeat_index * element_count;
        auto probe = probes.read(pixel_index);
        auto ray = make_ray(make_float3(probe.x, probe.y, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f),
                            0.0f, 2.0f);
        auto closest = accel.intersect_motion(ray, probe.z, {});
        auto any = accel.intersect_any_motion(ray, probe.z, {});
        UInt user_id = ~0u;
        $if (!closest->miss()) {
            user_id = accel.instance_user_id(closest->inst);
        };
        auto flags = cast<uint>(any) |
                     (cast<uint>(closest->is_triangle()) << 1u);
        summaries.write(result_index, make_uint4(
                                          closest->inst,
                                          closest->prim,
                                          user_id,
                                          flags));
        details.write(result_index, make_float4(
                                        closest->distance(),
                                        closest->bary.x,
                                        closest->bary.y,
                                        cast<float>(closest->is_triangle())));
    };

    auto shader = device.compile(trace);
    auto probe_buffer = device.create_buffer<float4>(element_count);
    auto summary_buffer = device.create_buffer<uint4>(result_count);
    auto detail_buffer = device.create_buffer<float4>(result_count);
    luisa::vector<uint4> host_summaries(result_count);
    luisa::vector<float4> host_details(result_count);

    auto run = [&](const ProbeSet &probe_set, Accel &accel,
                   uint32_t expected_user_id, luisa::string_view phase) {
        stream << probe_buffer.copy_from(luisa::span{probe_set.probes});
        for (auto repeat = 0u; repeat < repeat_count; repeat++) {
            stream << shader(accel, probe_buffer, summary_buffer, detail_buffer,
                             repeat)
                          .dispatch(dispatch_width, dispatch_height);
        }
        stream << summary_buffer.copy_to(luisa::span{host_summaries})
               << detail_buffer.copy_to(luisa::span{host_details})
               << synchronize();
        check_results(phase, luisa::span{host_summaries},
                      luisa::span{host_details},
                      luisa::span{probe_set.barycentrics}, expected_user_id);
    };

    auto identity_probes = make_probes(false);
    run(identity_probes, identity_accel, identity_user_id,
        "identity-key SRT motion");

    auto translating_probes = make_probes(true);
    run(translating_probes, translating_accel, translating_user_id,
        "translating SRT motion at t=0.25/0.75");
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    "HIP direct motion traversal is stable across 16x16 workgroup rows"_test = [&] {
        test_hip_motion_workgroup_rows(dc->device);
    };
}
