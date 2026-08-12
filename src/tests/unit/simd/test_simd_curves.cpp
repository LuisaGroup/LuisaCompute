// Exact SIMD curve packet coverage.
//
// The five-ray dispatch exercises direct closest/any traversal, query-all and
// query-any, opaque auto-commit, non-opaque accept/reject handlers, curve hit
// classification, all four curve bases, and an inactive tail at every
// supported SIMD width.

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

constexpr auto curve_basis = CurveBasis::PIECEWISE_LINEAR;
constexpr auto test_dispatch_size = 8u;
constexpr auto motion_dispatch_size = 5u;

void expect_near(
    float actual, float expected,
    luisa::string_view label) {
    expect(std::abs(actual - expected) <= 2.0e-4f)
        << luisa::format(
               "{}: got {}, expected {}", label, actual, expected);
}

void check_query_result(
    const uint4 &summary, const float4 &committed,
    const float4 &candidate, uint32_t test_case,
    uint32_t expected_instance, uint32_t expected_primitive,
    float expected_distance,
    luisa::string_view label) {
    auto surface = static_cast<uint32_t>(HitType::Surface);
    if (test_case == 0u) {
        expect(static_cast<bool>(
            all(summary == make_uint4(
                               surface, expected_instance,
                               expected_primitive, 0u))))
            << luisa::format("{} opaque summary mismatch", label);
        expect_near(
            committed.x, expected_distance,
            "opaque committed distance");
        expect_near(committed.y, 0.5f, "opaque committed parameter");
        expect_near(committed.z, -1.0f, "opaque committed marker");
        expect_near(committed.w, 1.0f, "opaque committed classification");
        expect_near(candidate.x, -1.0f, "opaque handler sentinel");
        return;
    }
    if (test_case == 1u) {
        expect(static_cast<bool>(
            all(summary == make_uint4(
                               surface, expected_instance,
                               expected_primitive, 1u))))
            << luisa::format("{} accepted summary mismatch", label);
        expect_near(
            committed.x, expected_distance,
            "accepted committed distance");
        expect_near(committed.y, 0.5f, "accepted committed parameter");
        expect_near(committed.z, -1.0f, "accepted committed marker");
        expect_near(committed.w, 1.0f, "accepted committed classification");
        expect_near(
            candidate.x, expected_distance,
            "accepted candidate distance");
        expect_near(candidate.y, 0.5f, "accepted candidate parameter");
        expect_near(candidate.z, -1.0f, "accepted candidate marker");
        expect_near(candidate.w, 1.0f, "accepted candidate classification");
        return;
    }
    expect(static_cast<bool>(
        all(summary == make_uint4(
                           static_cast<uint32_t>(HitType::Miss),
                           ~0u, ~0u, 1u))))
        << luisa::format("{} rejected summary mismatch", label);
    expect_near(candidate.x, 0.40f, "rejected candidate distance");
    expect_near(candidate.y, 0.5f, "rejected candidate parameter");
    expect_near(candidate.z, -1.0f, "rejected candidate marker");
    expect_near(candidate.w, 1.0f, "rejected candidate classification");
}

void test_motion_curves(
    Device &device, Stream &stream,
    uint32_t width, Curve &static_curve) {
    const std::array motion_control_points{
        // Primitive-motion keyframe zero.
        make_float4(-0.5f, 0.0f, 0.0f, 0.1f),
        make_float4(0.5f, 0.0f, 0.0f, 0.1f),
        // Primitive-motion keyframe one moves 0.4 units away from the ray.
        make_float4(-0.5f, 0.0f, -0.4f, 0.1f),
        make_float4(0.5f, 0.0f, -0.4f, 0.1f)};
    const std::array segments{0u};
    auto control_point_buffer =
        device.create_buffer<float4>(motion_control_points.size());
    auto segment_buffer = device.create_buffer<uint>(segments.size());

    AccelOption primitive_motion_option{};
    primitive_motion_option.motion.keyframe_count = 2u;
    primitive_motion_option.motion.time_start = 0.0f;
    primitive_motion_option.motion.time_end = 1.0f;
    auto motion_curve = device.create_curve(
        curve_basis, control_point_buffer, segment_buffer,
        primitive_motion_option);
    auto primitive_motion_accel = device.create_accel();
    primitive_motion_accel.emplace_back(
        motion_curve, make_float4x4(1.0f), 0xffu, false);

    AccelMotionOption instance_motion_option{};
    instance_motion_option.keyframe_count = 2u;
    instance_motion_option.time_start = 0.0f;
    instance_motion_option.time_end = 1.0f;
    instance_motion_option.mode = AccelMotionMode::MATRIX;
    auto motion_instance = device.create_motion_instance(
        static_curve, instance_motion_option);
    const std::array instance_keyframes{
        make_float4x4(1.0f),
        translation(make_float3(0.0f, 1.0f, 0.0f))};
    motion_instance.set_keyframes(luisa::span{instance_keyframes});
    auto instance_motion_accel = device.create_accel();
    instance_motion_accel.emplace_back(
        motion_instance, make_float4x4(1.0f), 0xffu, false);

    Kernel1D kernel = [width](
                          AccelVar primitive_scene,
                          AccelVar instance_scene,
                          BufferUInt4 primitive_summary,
                          BufferFloat4 primitive_detail,
                          BufferUInt4 instance_summary,
                          BufferFloat4 instance_detail) noexcept {
        set_block_size(32u, 1u, 1u);
        set_warp_size(static_cast<uint8_t>(width));
        auto index = dispatch_x();
        auto time = 0.5f * cast<float>(index % 3u);
        auto options = AccelTraceOptions{
            .curve_bases = {CurveBasis::PIECEWISE_LINEAR}};

        auto primitive_ray = make_ray(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -2.0f),
            0.0f, 1.0f);
        auto primitive_direct = primitive_scene.intersect_motion(
            primitive_ray, time, options);
        auto primitive_any = primitive_scene.intersect_any_motion(
            primitive_ray, time, options);
        UInt primitive_callback_count = 0u;
        auto primitive_query =
            primitive_scene.traverse_motion(primitive_ray, time, options)
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        primitive_callback_count += 1u;
                        candidate.commit();
                    })
                .on_procedural_candidate(
                    [](ProceduralCandidate &) noexcept {})
                .trace();
        primitive_summary.write(
            index,
            make_uint4(
                primitive_direct->inst, primitive_direct->prim,
                cast<uint>(primitive_any), primitive_callback_count));
        primitive_detail.write(
            index,
            make_float4(
                primitive_direct->distance(), primitive_direct->bary.y,
                primitive_query->distance(), primitive_query->bary.y));

        auto instance_ray = make_ray(
            make_float3(0.0f, time, 1.0f),
            make_float3(0.0f, 0.0f, -2.0f),
            0.0f, 1.0f);
        auto instance_direct = instance_scene.intersect_motion(
            instance_ray, time, options);
        auto instance_any = instance_scene.intersect_any_motion(
            instance_ray, time, options);
        UInt instance_callback_count = 0u;
        auto instance_query =
            instance_scene.traverse_motion(instance_ray, time, options)
                .on_surface_candidate(
                    [&](SurfaceCandidate &candidate) noexcept {
                        instance_callback_count += 1u;
                        candidate.commit();
                    })
                .on_procedural_candidate(
                    [](ProceduralCandidate &) noexcept {})
                .trace();
        instance_summary.write(
            index,
            make_uint4(
                instance_direct->inst, instance_direct->prim,
                cast<uint>(instance_any), instance_callback_count));
        instance_detail.write(
            index,
            make_float4(
                instance_direct->distance(), instance_direct->bary.y,
                instance_query->distance(), instance_query->bary.y));
    };
    auto shader = device.compile(kernel);
    auto primitive_summary =
        device.create_buffer<uint4>(motion_dispatch_size);
    auto primitive_detail =
        device.create_buffer<float4>(motion_dispatch_size);
    auto instance_summary =
        device.create_buffer<uint4>(motion_dispatch_size);
    auto instance_detail =
        device.create_buffer<float4>(motion_dispatch_size);
    std::array<uint4, motion_dispatch_size> host_primitive_summary{};
    std::array<float4, motion_dispatch_size> host_primitive_detail{};
    std::array<uint4, motion_dispatch_size> host_instance_summary{};
    std::array<float4, motion_dispatch_size> host_instance_detail{};

    stream << control_point_buffer.copy_from(
                  luisa::span{motion_control_points})
           << segment_buffer.copy_from(luisa::span{segments})
           << motion_curve.build()
           << motion_instance.build()
           << primitive_motion_accel.build()
           << instance_motion_accel.build()
           << shader(
                  primitive_motion_accel, instance_motion_accel,
                  primitive_summary, primitive_detail,
                  instance_summary, instance_detail)
                  .dispatch(motion_dispatch_size)
           << primitive_summary.copy_to(
                  luisa::span{host_primitive_summary})
           << primitive_detail.copy_to(
                  luisa::span{host_primitive_detail})
           << instance_summary.copy_to(
                  luisa::span{host_instance_summary})
           << instance_detail.copy_to(
                  luisa::span{host_instance_detail})
           << synchronize();

    for (auto i = 0u; i < motion_dispatch_size; i++) {
        auto time = 0.5f * static_cast<float>(i % 3u);
        auto expected_primitive_t = 0.45f + 0.2f * time;
        expect(static_cast<bool>(
            all(host_primitive_summary[i] == make_uint4(0u, 0u, 1u, 1u))))
            << "primitive-motion curve summary mismatch";
        expect_near(
            host_primitive_detail[i].x, expected_primitive_t,
            "primitive-motion direct distance");
        expect_near(
            host_primitive_detail[i].y, -1.0f,
            "primitive-motion direct marker");
        expect_near(
            host_primitive_detail[i].z, expected_primitive_t,
            "primitive-motion query distance");
        expect_near(
            host_primitive_detail[i].w, -1.0f,
            "primitive-motion query marker");

        expect(static_cast<bool>(
            all(host_instance_summary[i] == make_uint4(0u, 0u, 1u, 1u))))
            << "motion-instance curve summary mismatch";
        expect_near(
            host_instance_detail[i].x, 0.45f,
            "motion-instance direct distance");
        expect_near(
            host_instance_detail[i].y, -1.0f,
            "motion-instance direct marker");
        expect_near(
            host_instance_detail[i].z, 0.45f,
            "motion-instance query distance");
        expect_near(
            host_instance_detail[i].w, -1.0f,
            "motion-instance query marker");
    }
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    Context context{argc > 0 ? argv[0] : ""};

    const std::array control_points{
        make_float4(-0.5f, 0.0f, 0.0f, 0.1f),
        make_float4(0.5f, 0.0f, 0.0f, 0.1f),
        make_float4(-0.5f, 1.0f, 0.1f, 0.1f),
        make_float4(0.5f, 1.0f, 0.1f, 0.1f)};
    const std::array segments{0u, 2u};
    const std::array cubic_control_points{
        make_float4(-1.5f, 0.0f, 0.0f, 0.1f),
        make_float4(-0.5f, 0.0f, 0.0f, 0.1f),
        make_float4(0.5f, 0.0f, 0.0f, 0.1f),
        make_float4(1.5f, 0.0f, 0.0f, 0.1f)};
    const std::array cubic_segments{0u};

    for (auto width : std::array{1u, 2u, 4u, 8u, 16u}) {
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(width);
        auto device = context.create_device("simd", &config);
        auto stream = device.create_stream();

        auto control_point_buffer =
            device.create_buffer<float4>(control_points.size());
        auto segment_buffer =
            device.create_buffer<uint>(segments.size());
        auto cubic_control_point_buffer =
            device.create_buffer<float4>(cubic_control_points.size());
        auto cubic_segment_buffer =
            device.create_buffer<uint>(cubic_segments.size());
        auto curve = device.create_curve(
            curve_basis, control_point_buffer, segment_buffer);
        auto bspline = device.create_curve(
            CurveBasis::CUBIC_BSPLINE,
            cubic_control_point_buffer, cubic_segment_buffer);
        auto catmull_rom = device.create_curve(
            CurveBasis::CATMULL_ROM,
            cubic_control_point_buffer, cubic_segment_buffer);
        auto bezier = device.create_curve(
            CurveBasis::BEZIER,
            cubic_control_point_buffer, cubic_segment_buffer);
        auto accel = device.create_accel();
        accel.emplace_back(
            curve, translation(make_float3(-2.0f, 0.0f, 0.0f)),
            0xffu, true);
        accel.emplace_back(
            curve, make_float4x4(1.0f), 0xffu, false);
        accel.emplace_back(
            bspline, translation(make_float3(2.0f, 0.0f, 0.0f)),
            0xffu, true);
        accel.emplace_back(
            catmull_rom, translation(make_float3(4.0f, 0.0f, 0.0f)),
            0xffu, true);
        accel.emplace_back(
            bezier, translation(make_float3(6.0f, 0.0f, 0.0f)),
            0xffu, true);

        Kernel1D kernel = [width](
                              AccelVar scene,
                              BufferUInt4 direct_summary,
                              BufferFloat4 direct_details,
                              BufferUInt4 all_summary,
                              BufferFloat4 all_committed,
                              BufferFloat4 all_candidate,
                              BufferUInt4 any_summary,
                              BufferFloat4 any_committed,
                              BufferFloat4 any_candidate) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto basis_probe = index >= 5u;
            auto test_case = ite(basis_probe, 0u, index % 3u);
            auto origin_x = ite(
                basis_probe,
                2.0f * cast<float>(index - 4u),
                ite(test_case == 0u, -2.0f, 0.0f));
            auto origin_y = ite(test_case == 2u, 1.0f, 0.0f);
            auto ray = make_ray(
                make_float3(origin_x, origin_y, 1.0f),
                make_float3(0.0f, 0.0f, -2.0f),
                0.0f, 1.0f);
            auto options = AccelTraceOptions{
                .curve_bases = {
                    CurveBasis::PIECEWISE_LINEAR,
                    CurveBasis::CUBIC_BSPLINE,
                    CurveBasis::CATMULL_ROM,
                    CurveBasis::BEZIER}};

            auto direct = scene.intersect(ray, options);
            auto occluded = scene.intersect_any(ray, options);
            direct_summary.write(
                index,
                make_uint4(
                    direct->inst, direct->prim,
                    cast<uint>(occluded),
                    cast<uint>(direct->is_curve())));
            direct_details.write(
                index,
                make_float4(
                    direct->distance(), direct->curve_parameter(),
                    direct->bary.y, 0.0f));

            UInt all_callback_count = 0u;
            Float4 all_seen = make_float4(-1.0f);
            auto all = scene.traverse(ray, options)
                           .on_surface_candidate(
                               [&](SurfaceCandidate &candidate) noexcept {
                                   auto hit = candidate.hit();
                                   all_callback_count += 1u;
                                   all_seen = make_float4(
                                       hit->distance(),
                                       hit->curve_parameter(),
                                       hit->bary.y,
                                       cast<float>(hit->is_curve()));
                                   $if (test_case == 1u) {
                                       candidate.commit();
                                   };
                               })
                           .on_procedural_candidate(
                               [](ProceduralCandidate &) noexcept {})
                           .trace();
            all_summary.write(
                index,
                make_uint4(
                    all->hit_type, all->inst,
                    all->prim, all_callback_count));
            all_committed.write(
                index,
                make_float4(
                    all->distance(), all->curve_parameter(),
                    all->bary.y, cast<float>(all->is_curve())));
            all_candidate.write(index, all_seen);

            UInt any_callback_count = 0u;
            Float4 any_seen = make_float4(-1.0f);
            auto any = scene.traverse_any(ray, options)
                           .on_surface_candidate(
                               [&](SurfaceCandidate &candidate) noexcept {
                                   auto hit = candidate.hit();
                                   any_callback_count += 1u;
                                   any_seen = make_float4(
                                       hit->distance(),
                                       hit->curve_parameter(),
                                       hit->bary.y,
                                       cast<float>(hit->is_curve()));
                                   $if (test_case == 1u) {
                                       candidate.commit();
                                   };
                               })
                           .on_procedural_candidate(
                               [](ProceduralCandidate &) noexcept {})
                           .trace();
            any_summary.write(
                index,
                make_uint4(
                    any->hit_type, any->inst,
                    any->prim, any_callback_count));
            any_committed.write(
                index,
                make_float4(
                    any->distance(), any->curve_parameter(),
                    any->bary.y, cast<float>(any->is_curve())));
            any_candidate.write(index, any_seen);
        };
        auto shader = device.compile(kernel);

        auto direct_summary = device.create_buffer<uint4>(test_dispatch_size);
        auto direct_details = device.create_buffer<float4>(test_dispatch_size);
        auto all_summary = device.create_buffer<uint4>(test_dispatch_size);
        auto all_committed = device.create_buffer<float4>(test_dispatch_size);
        auto all_candidate = device.create_buffer<float4>(test_dispatch_size);
        auto any_summary = device.create_buffer<uint4>(test_dispatch_size);
        auto any_committed = device.create_buffer<float4>(test_dispatch_size);
        auto any_candidate = device.create_buffer<float4>(test_dispatch_size);
        std::array<uint4, test_dispatch_size> host_direct_summary{};
        std::array<float4, test_dispatch_size> host_direct_details{};
        std::array<uint4, test_dispatch_size> host_all_summary{};
        std::array<float4, test_dispatch_size> host_all_committed{};
        std::array<float4, test_dispatch_size> host_all_candidate{};
        std::array<uint4, test_dispatch_size> host_any_summary{};
        std::array<float4, test_dispatch_size> host_any_committed{};
        std::array<float4, test_dispatch_size> host_any_candidate{};

        stream << control_point_buffer.copy_from(
                      luisa::span{control_points})
               << segment_buffer.copy_from(luisa::span{segments})
               << cubic_control_point_buffer.copy_from(
                      luisa::span{cubic_control_points})
               << cubic_segment_buffer.copy_from(
                      luisa::span{cubic_segments})
               << curve.build()
               << bspline.build()
               << catmull_rom.build()
               << bezier.build()
               << accel.build()
               << shader(
                      accel, direct_summary, direct_details,
                      all_summary, all_committed, all_candidate,
                      any_summary, any_committed, any_candidate)
                      .dispatch(test_dispatch_size)
               << direct_summary.copy_to(
                      luisa::span{host_direct_summary})
               << direct_details.copy_to(
                      luisa::span{host_direct_details})
               << all_summary.copy_to(luisa::span{host_all_summary})
               << all_committed.copy_to(luisa::span{host_all_committed})
               << all_candidate.copy_to(luisa::span{host_all_candidate})
               << any_summary.copy_to(luisa::span{host_any_summary})
               << any_committed.copy_to(luisa::span{host_any_committed})
               << any_candidate.copy_to(luisa::span{host_any_candidate})
               << synchronize();

        for (auto i = 0u; i < test_dispatch_size; i++) {
            auto basis_probe = i >= 5u;
            auto test_case = basis_probe ? 0u : i % 3u;
            auto expected_instance = basis_probe     ? i - 3u :
                                     test_case == 0u ? 0u :
                                                       1u;
            auto expected_primitive =
                !basis_probe && test_case == 2u ? 1u : 0u;
            auto expected_t =
                !basis_probe && test_case == 2u ? 0.40f : 0.45f;
            expect(static_cast<bool>(
                all(host_direct_summary[i] ==
                    make_uint4(
                        expected_instance, expected_primitive,
                        1u, 1u))))
                << "direct curve packet summary mismatch";
            expect_near(
                host_direct_details[i].x, expected_t,
                "direct curve distance");
            expect_near(
                host_direct_details[i].y, 0.5f,
                "direct curve parameter");
            expect_near(
                host_direct_details[i].z, -1.0f,
                "direct curve marker");
            check_query_result(
                host_all_summary[i], host_all_committed[i],
                host_all_candidate[i], test_case,
                expected_instance, expected_primitive,
                expected_t, "query-all");
            check_query_result(
                host_any_summary[i], host_any_committed[i],
                host_any_candidate[i], test_case,
                expected_instance, expected_primitive,
                expected_t, "query-any");
        }
        test_motion_curves(device, stream, width, curve);
    }
    return 0;
}
