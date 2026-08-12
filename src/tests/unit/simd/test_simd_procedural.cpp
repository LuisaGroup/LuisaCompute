// Exact SIMD procedural-primitive and ray-query packet coverage.
//
// This exercises W1/W2/W4/W8/W16, inactive tails, direct-trace rejection,
// query-all/query-any commit/reject/terminate behavior, ordered continuation
// scans, mixed triangle/procedural traversal, visibility, primitive motion,
// motion instances, and mesh/procedural provider refresh after accel rebuild.

#include "ut/ut.hpp"

#include <array>
#include <cmath>

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto dispatch_size = 35u;
constexpr auto long_candidate_count = 40u;

void expect_near(
    float actual, float expected,
    luisa::string_view label) {
    expect(std::abs(actual - expected) <= 1.0e-5f)
        << luisa::format(
               "{}: got {}, expected {}", label, actual, expected);
}

void test_static_queries(
    Device &device, Stream &stream, uint32_t width) {
    const std::array boxes{
        AABB{.packed_min = {-0.5f, -0.5f, -0.1f},
             .packed_max = {0.5f, 0.5f, 0.1f}},
        AABB{.packed_min = {-0.5f, -0.5f, -0.1f},
             .packed_max = {0.5f, 0.5f, 0.1f}},
        AABB{.packed_min = {-0.5f, -0.5f, -0.1f},
             .packed_max = {0.5f, 0.5f, 0.1f}}};
    auto box_buffer = device.create_buffer<AABB>(boxes.size());
    auto procedural = device.create_procedural_primitive(box_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(
        procedural, make_float4x4(1.0f), 0x1u, 17u);

    Kernel1D kernel = [width](
                          AccelVar scene,
                          BufferUInt4 direct_summary,
                          BufferUInt4 all_summary,
                          BufferFloat2 all_detail,
                          BufferUInt4 any_summary,
                          BufferFloat2 any_detail) noexcept {
        set_block_size(32u, 1u, 1u);
        set_warp_size(static_cast<uint8_t>(width));
        auto index = dispatch_x();
        auto test_case = index % 5u;
        auto origin_x = ite(test_case == 4u, 2.0f, 0.0f);
        auto visibility = ite(test_case == 3u, 0x2u, 0x1u);
        auto ray = make_ray(
            make_float3(origin_x, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0.0f, 4.0f);
        auto options = AccelTraceOptions{
            .visibility_mask = visibility};

        auto direct = scene.intersect(ray, options);
        auto direct_any = scene.intersect_any(ray, options);
        direct_summary.write(
            index,
            make_uint4(
                direct->inst, direct->prim,
                cast<uint>(direct_any), 0u));

        UInt all_count = 0u;
        UInt all_order = 0u;
        Float all_tmax = ray->t_max();
        auto all = scene.traverse(ray, options)
                       .on_surface_candidate(
                           [](SurfaceCandidate &) noexcept {})
                       .on_procedural_candidate(
                           [&](ProceduralCandidate &candidate) noexcept {
                               auto hit = candidate.hit();
                               all_count += 1u;
                               all_order = all_order * 4u + hit->prim + 1u;
                               $if ((test_case == 0u) &
                                    (hit->prim == 1u)) {
                                   candidate.commit(0.95f);
                               };
                               $if (test_case == 1u) {
                                   $if (hit->prim == 0u) {
                                       candidate.commit(-1.0f);
                                   };
                                   $if (hit->prim == 1u) {
                                       candidate.commit(5.0f);
                                   };
                                   $if (hit->prim == 2u) {
                                       candidate.commit(0.90f);
                                   };
                               };
                               $if ((test_case == 2u) &
                                    (hit->prim == 1u)) {
                                   candidate.terminate();
                               };
                               all_tmax = candidate.ray()->t_max();
                           })
                       .trace();
        all_summary.write(
            index,
            make_uint4(
                all->hit_type, all->inst,
                all->prim, all_count));
        all_detail.write(
            index,
            make_float2(
                all->distance(),
                cast<float>(all_order) + all_tmax * 0.001f));

        UInt any_count = 0u;
        UInt any_order = 0u;
        Float any_tmax = ray->t_max();
        auto any = scene.traverse_any(ray, options)
                       .on_surface_candidate(
                           [](SurfaceCandidate &) noexcept {})
                       .on_procedural_candidate(
                           [&](ProceduralCandidate &candidate) noexcept {
                               auto hit = candidate.hit();
                               any_count += 1u;
                               any_order = any_order * 4u + hit->prim + 1u;
                               $if ((test_case == 0u) &
                                    (hit->prim == 1u)) {
                                   candidate.commit(0.95f);
                               };
                               $if (test_case == 1u) {
                                   $if (hit->prim == 0u) {
                                       candidate.commit(-1.0f);
                                   };
                                   $if (hit->prim == 1u) {
                                       candidate.commit(5.0f);
                                   };
                                   $if (hit->prim == 2u) {
                                       candidate.commit(0.90f);
                                   };
                               };
                               $if ((test_case == 2u) &
                                    (hit->prim == 1u)) {
                                   candidate.terminate();
                               };
                               any_tmax = candidate.ray()->t_max();
                           })
                       .trace();
        any_summary.write(
            index,
            make_uint4(
                any->hit_type, any->inst,
                any->prim, any_count));
        any_detail.write(
            index,
            make_float2(
                any->distance(),
                cast<float>(any_order) + any_tmax * 0.001f));
    };

    auto shader = device.compile(kernel);
    auto direct_summary = device.create_buffer<uint4>(dispatch_size);
    auto all_summary = device.create_buffer<uint4>(dispatch_size);
    auto all_detail = device.create_buffer<float2>(dispatch_size);
    auto any_summary = device.create_buffer<uint4>(dispatch_size);
    auto any_detail = device.create_buffer<float2>(dispatch_size);
    std::array<uint4, dispatch_size> host_direct{};
    std::array<uint4, dispatch_size> host_all{};
    std::array<float2, dispatch_size> host_all_detail{};
    std::array<uint4, dispatch_size> host_any{};
    std::array<float2, dispatch_size> host_any_detail{};

    stream << box_buffer.copy_from(luisa::span{boxes})
           << procedural.build()
           << accel.build()
           << shader(
                  accel, direct_summary,
                  all_summary, all_detail,
                  any_summary, any_detail)
                  .dispatch(dispatch_size)
           << direct_summary.copy_to(luisa::span{host_direct})
           << all_summary.copy_to(luisa::span{host_all})
           << all_detail.copy_to(luisa::span{host_all_detail})
           << any_summary.copy_to(luisa::span{host_any})
           << any_detail.copy_to(luisa::span{host_any_detail})
           << synchronize();

    constexpr auto miss = static_cast<uint32_t>(HitType::Miss);
    constexpr auto procedural_hit =
        static_cast<uint32_t>(HitType::Procedural);
    for (auto i = 0u; i < dispatch_size; i++) {
        auto test_case = i % 5u;
        expect(static_cast<bool>(
            all(host_direct[i] == make_uint4(~0u, ~0u, 0u, 0u))))
            << "direct traversal must reject procedural AABBs";
        if (test_case == 0u) {
            expect(static_cast<bool>(
                all(host_all[i] ==
                    make_uint4(procedural_hit, 0u, 1u, 3u))));
            expect(static_cast<bool>(
                all(host_any[i] ==
                    make_uint4(procedural_hit, 0u, 1u, 2u))));
            expect_near(host_all_detail[i].x, 0.95f, "query-all t");
            expect_near(host_any_detail[i].x, 0.95f, "query-any t");
            expect_near(host_all_detail[i].y, 27.00095f, "query-all order");
            expect_near(host_any_detail[i].y, 6.00095f, "query-any order");
        } else if (test_case == 1u) {
            expect(static_cast<bool>(
                all(host_all[i] ==
                    make_uint4(procedural_hit, 0u, 2u, 3u))));
            expect(static_cast<bool>(
                all(host_any[i] ==
                    make_uint4(procedural_hit, 0u, 2u, 3u))));
            expect_near(host_all_detail[i].x, 0.90f, "bounded query-all t");
            expect_near(host_any_detail[i].x, 0.90f, "bounded query-any t");
            expect_near(host_all_detail[i].y, 27.00090f, "bounded all order");
            expect_near(host_any_detail[i].y, 27.00090f, "bounded any order");
        } else if (test_case == 2u) {
            expect(static_cast<bool>(
                all(host_all[i] == make_uint4(miss, ~0u, ~0u, 2u))));
            expect(static_cast<bool>(
                all(host_any[i] == make_uint4(miss, ~0u, ~0u, 2u))));
            expect_near(host_all_detail[i].y, 6.004f, "terminated all order");
            expect_near(host_any_detail[i].y, 6.004f, "terminated any order");
        } else if (test_case == 3u) {
            expect(static_cast<bool>(
                all(host_all[i] == make_uint4(miss, ~0u, ~0u, 0u))));
            expect(static_cast<bool>(
                all(host_any[i] == make_uint4(miss, ~0u, ~0u, 0u))))
                << luisa::format(
                       "query-any miss summary mismatch: case {}, "
                       "got ({}, {}, {}, {})",
                       test_case, host_any[i].x, host_any[i].y,
                       host_any[i].z, host_any[i].w);
        } else {
            // Embree's occlusion traversal may conservatively invoke a user
            // primitive callback for rays that do not intersect the exact
            // AABB. The handler rejects those candidates, so only the final
            // miss and bounded callback count are contractual here.
            expect(static_cast<bool>(
                all(host_all[i] == make_uint4(miss, ~0u, ~0u, 0u))));
            expect(host_any[i].x == miss &&
                   host_any[i].y == ~0u &&
                   host_any[i].z == ~0u &&
                   host_any[i].w <= boxes.size())
                << "conservative query-any procedural miss mismatch";
        }
    }
}

void test_long_candidate_chain(
    Device &device, Stream &stream, uint32_t width) {
    std::array<AABB, long_candidate_count> boxes{};
    for (auto &box : boxes) {
        box = AABB{
            .packed_min = {-0.5f, -0.5f, -0.1f},
            .packed_max = {0.5f, 0.5f, 0.1f}};
    }
    auto box_buffer = device.create_buffer<AABB>(boxes.size());
    auto procedural = device.create_procedural_primitive(box_buffer);
    auto accel = device.create_accel();
    accel.emplace_back(procedural);
    auto result = device.create_buffer<uint4>(1u);
    auto distance = device.create_buffer<float>(1u);
    Kernel1D kernel = [width](
                          AccelVar scene, BufferUInt4 result,
                          BufferFloat distance) noexcept {
        set_block_size(32u, 1u, 1u);
        set_warp_size(static_cast<uint8_t>(width));
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
        UInt count = 0u;
        UInt checksum = 0u;
        auto hit = scene.traverse(ray, {})
                       .on_surface_candidate(
                           [](SurfaceCandidate &) noexcept {})
                       .on_procedural_candidate(
                           [&](ProceduralCandidate &candidate) noexcept {
                               auto candidate_hit = candidate.hit();
                               count += 1u;
                               checksum += candidate_hit->prim + 1u;
                               $if (candidate_hit->prim ==
                                    long_candidate_count - 1u) {
                                   candidate.commit(0.95f);
                               };
                           })
                       .trace();
        result.write(
            0u, make_uint4(hit->hit_type, hit->inst, count, checksum));
        distance.write(0u, hit->distance());
    };
    auto shader = device.compile(kernel);
    uint4 host_result{};
    float host_distance{};
    stream << box_buffer.copy_from(luisa::span{boxes})
           << procedural.build()
           << accel.build()
           << shader(accel, result, distance).dispatch(1u)
           << result.copy_to(&host_result)
           << distance.copy_to(&host_distance)
           << synchronize();
    expect(static_cast<bool>(
        all(host_result ==
            make_uint4(
                static_cast<uint32_t>(HitType::Procedural),
                0u, long_candidate_count,
                long_candidate_count * (long_candidate_count + 1u) / 2u))))
        << "long procedural continuation chain mismatch";
    expect_near(host_distance, 0.95f, "long-chain distance");
}

void test_procedural_summary_refresh(
    Device &device, Stream &stream, uint32_t width) {
    const std::array vertices{
        make_float3(-0.5f, -0.5f, 0.0f),
        make_float3(0.5f, -0.5f, 0.0f),
        make_float3(0.0f, 0.5f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    const std::array boxes{
        AABB{.packed_min = {-0.5f, -0.5f, -0.1f},
             .packed_max = {0.5f, 0.5f, 0.1f}}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(triangles.size());
    auto box_buffer = device.create_buffer<AABB>(boxes.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto procedural = device.create_procedural_primitive(box_buffer);
    auto accel = device.create_accel({.allow_update = true});
    accel.emplace_back(
        mesh, make_float4x4(1.0f), 0xffu, false);

    Kernel1D kernel = [width](
                          AccelVar scene,
                          BufferUInt4 result) noexcept {
        set_block_size(32u, 1u, 1u);
        set_warp_size(static_cast<uint8_t>(width));
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
        UInt callback_mask = 0u;
        auto hit = scene.traverse(ray, {})
                       .on_surface_candidate(
                           [&](SurfaceCandidate &candidate) noexcept {
                               callback_mask |= 1u;
                               candidate.commit();
                           })
                       .on_procedural_candidate(
                           [&](ProceduralCandidate &candidate) noexcept {
                               callback_mask |= 2u;
                               candidate.commit(1.0f);
                           })
                       .trace();
        result.write(
            0u, make_uint4(
                    hit->hit_type, callback_mask,
                    hit->inst, hit->prim));
    };
    auto shader = device.compile(kernel);
    auto result = device.create_buffer<uint4>(1u);
    uint4 host_result{};
    auto check = [&](uint32_t expected_kind, uint32_t expected_mask,
                     luisa::string_view label) {
        stream << accel.build()
               << shader(accel, result).dispatch(1u)
               << result.copy_to(&host_result)
               << synchronize();
        expect(static_cast<bool>(
            all(host_result == make_uint4(
                                   expected_kind, expected_mask,
                                   0u, 0u))))
            << label;
    };

    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << box_buffer.copy_from(luisa::span{boxes})
           << mesh.build()
           << procedural.build()
           << synchronize();
    check(
        static_cast<uint32_t>(HitType::Surface), 1u,
        "mesh must select the triangle-only ray-query provider");
    accel.set_procedural_primitive(0u, procedural);
    check(
        static_cast<uint32_t>(HitType::Procedural), 2u,
        "procedural replacement must restore the generic provider");
    accel.set_mesh(0u, mesh);
    check(
        static_cast<uint32_t>(HitType::Surface), 1u,
        "mesh replacement must restore the triangle-only provider");
}

void test_mixed_surface_and_motion(
    Device &device, Stream &stream, uint32_t width) {
    const std::array vertices{
        make_float3(-0.5f, -0.5f, 0.0f),
        make_float3(0.5f, -0.5f, 0.0f),
        make_float3(0.0f, 0.5f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    const std::array static_box{
        AABB{.packed_min = {-0.5f, -0.5f, 0.4f},
             .packed_max = {0.5f, 0.5f, 0.6f}}};
    const std::array motion_boxes{
        AABB{.packed_min = {-0.5f, -0.5f, -0.1f},
             .packed_max = {0.5f, 0.5f, 0.1f}},
        AABB{.packed_min = {-0.5f, -0.5f, -2.1f},
             .packed_max = {0.5f, 0.5f, -1.9f}}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto static_box_buffer = device.create_buffer<AABB>(static_box.size());
    auto motion_box_buffer =
        device.create_buffer<AABB>(motion_boxes.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto static_procedural =
        device.create_procedural_primitive(static_box_buffer);
    AccelOption primitive_motion_option{};
    primitive_motion_option.motion.keyframe_count = 2u;
    primitive_motion_option.motion.time_start = 0.0f;
    primitive_motion_option.motion.time_end = 1.0f;
    auto motion_procedural = device.create_procedural_primitive(
        motion_box_buffer, primitive_motion_option);

    auto mixed_accel = device.create_accel();
    mixed_accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    mixed_accel.emplace_back(static_procedural);
    auto primitive_motion_accel = device.create_accel();
    primitive_motion_accel.emplace_back(motion_procedural);

    AccelMotionOption instance_option{};
    instance_option.keyframe_count = 2u;
    instance_option.time_start = 0.0f;
    instance_option.time_end = 1.0f;
    instance_option.mode = AccelMotionMode::MATRIX;
    auto motion_instance = device.create_motion_instance(
        static_procedural, instance_option);
    const std::array instance_keys{
        translation(make_float3(0.0f, 0.0f, -0.5f)),
        translation(make_float3(0.0f, 0.0f, -2.5f))};
    motion_instance.set_keyframes(luisa::span{instance_keys});
    auto instance_motion_accel = device.create_accel();
    instance_motion_accel.emplace_back(motion_instance);

    auto mixed_result = device.create_buffer<uint4>(1u);
    auto mixed_distance = device.create_buffer<float>(1u);
    Kernel1D mixed_kernel = [width](
                                AccelVar scene, BufferUInt4 result,
                                BufferFloat distance) noexcept {
        set_block_size(32u, 1u, 1u);
        set_warp_size(static_cast<uint8_t>(width));
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 4.0f);
        UInt callback_mask = 0u;
        UInt callback_count = 0u;
        auto hit = scene.traverse(ray, {})
                       .on_surface_candidate(
                           [&](SurfaceCandidate &candidate) noexcept {
                               callback_mask |= 1u;
                               callback_count += 1u;
                               candidate.commit();
                           })
                       .on_procedural_candidate(
                           [&](ProceduralCandidate &candidate) noexcept {
                               callback_mask |= 2u;
                               callback_count += 1u;
                               candidate.commit(2.0f);
                           })
                       .trace();
        result.write(
            0u,
            make_uint4(
                hit->hit_type, hit->inst,
                callback_mask, callback_count));
        distance.write(0u, hit->distance());
    };

    auto motion_summary = device.create_buffer<uint4>(5u);
    auto motion_detail = device.create_buffer<float2>(5u);
    Kernel1D motion_kernel = [width](
                                 AccelVar primitive_scene,
                                 AccelVar instance_scene,
                                 BufferUInt4 summary,
                                 BufferFloat2 detail) noexcept {
        set_block_size(32u, 1u, 1u);
        set_warp_size(static_cast<uint8_t>(width));
        auto index = dispatch_x();
        auto time = 0.5f * cast<float>(index % 3u);
        auto ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 5.0f);
        UInt primitive_count = 0u;
        auto primitive_hit =
            primitive_scene.traverse_motion(ray, time, {})
                .on_surface_candidate(
                    [](SurfaceCandidate &) noexcept {})
                .on_procedural_candidate(
                    [&](ProceduralCandidate &candidate) noexcept {
                        primitive_count += 1u;
                        candidate.commit(1.0f + 2.0f * time);
                    })
                .trace();
        UInt instance_count = 0u;
        auto instance_hit =
            instance_scene.traverse_motion(ray, time, {})
                .on_surface_candidate(
                    [](SurfaceCandidate &) noexcept {})
                .on_procedural_candidate(
                    [&](ProceduralCandidate &candidate) noexcept {
                        instance_count += 1u;
                        candidate.commit(1.0f + 2.0f * time);
                    })
                .trace();
        summary.write(
            index,
            make_uint4(
                primitive_hit->hit_type, primitive_count,
                instance_hit->hit_type, instance_count));
        detail.write(
            index,
            make_float2(
                primitive_hit->distance(),
                instance_hit->distance()));
    };

    auto mixed_shader = device.compile(mixed_kernel);
    auto motion_shader = device.compile(motion_kernel);
    uint4 host_mixed{};
    float host_mixed_distance{};
    std::array<uint4, 5u> host_motion_summary{};
    std::array<float2, 5u> host_motion_detail{};
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << static_box_buffer.copy_from(luisa::span{static_box})
           << motion_box_buffer.copy_from(luisa::span{motion_boxes})
           << mesh.build()
           << static_procedural.build()
           << motion_procedural.build()
           << motion_instance.build()
           << mixed_accel.build()
           << primitive_motion_accel.build()
           << instance_motion_accel.build()
           << mixed_shader(
                  mixed_accel, mixed_result, mixed_distance)
                  .dispatch(1u)
           << motion_shader(
                  primitive_motion_accel, instance_motion_accel,
                  motion_summary, motion_detail)
                  .dispatch(5u)
           << mixed_result.copy_to(&host_mixed)
           << mixed_distance.copy_to(&host_mixed_distance)
           << motion_summary.copy_to(luisa::span{host_motion_summary})
           << motion_detail.copy_to(luisa::span{host_motion_detail})
           << synchronize();

    expect(static_cast<bool>(
        all(host_mixed ==
            make_uint4(
                static_cast<uint32_t>(HitType::Surface),
                0u, 3u, 2u))))
        << "mixed procedural/surface query mismatch";
    expect_near(host_mixed_distance, 1.0f, "mixed closest distance");
    for (auto i = 0u; i < 5u; i++) {
        auto expected = 1.0f + static_cast<float>(i % 3u);
        expect(static_cast<bool>(
            all(host_motion_summary[i] ==
                make_uint4(
                    static_cast<uint32_t>(HitType::Procedural), 1u,
                    static_cast<uint32_t>(HitType::Procedural), 1u))))
            << "procedural motion query summary mismatch";
        expect_near(
            host_motion_detail[i].x, expected,
            "primitive-motion procedural distance");
        expect_near(
            host_motion_detail[i].y, expected,
            "motion-instance procedural distance");
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    Context context{argc > 0 ? argv[0] : ""};
    for (auto width : std::array{1u, 2u, 4u, 8u, 16u}) {
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(width);
        auto device = context.create_device("simd", &config);
        auto stream = device.create_stream();
        test_static_queries(device, stream, width);
        test_long_candidate_chain(device, stream, width);
        test_procedural_summary_refresh(device, stream, width);
        test_mixed_surface_and_motion(device, stream, width);
    }
    return 0;
}
