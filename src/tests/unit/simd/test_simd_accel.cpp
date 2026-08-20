// Test for SIMD acceleration-structure packet execution.
// This test covers:
// - W1/W2/W4/W8/W16 closest-hit and any-hit traversal
// - varying-time motion closest-hit and any-hit traversal
// - MATRIX/SRT motion-instance keyframes, quaternion interpolation, outer
//   affine composition, device mutation, and packet traversal
// - non-opaque triangle ray-query rejection/commit under divergent handlers
// - varying device opacity mutation, opaque auto-commit, and inactive tails
// - direct varying and uniform instance metadata reads
// - host buffer-only metadata/grow/shrink updates and deferred BVH/query commit
// - divergent visibility masks, ray intervals, directions, and misses
// - uniform static and motion traces that must remain scalar within a packet
// - an inactive W16 tail with only three live lanes

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

constexpr auto thread_count = 35u;
constexpr auto deep_query_candidate_count = 35u;

[[nodiscard]] Ray make_host_ray(
    float3 origin, float3 direction,
    float t_min, float t_max) noexcept {
    return Ray{
        .compressed_origin = {origin.x, origin.y, origin.z},
        .compressed_t_min = t_min,
        .compressed_direction = {
            direction.x, direction.y, direction.z},
        .compressed_t_max = t_max,
    };
}

[[nodiscard]] MotionInstanceTransformSRT make_srt(
    float3 pivot, float4 quaternion, float3 scale,
    float3 shear, float3 translation) noexcept {
    return MotionInstanceTransformSRT{
        .pivot = {pivot.x, pivot.y, pivot.z},
        .quaternion = {
            quaternion.x, quaternion.y,
            quaternion.z, quaternion.w},
        .scale = {scale.x, scale.y, scale.z},
        .shear = {shear.x, shear.y, shear.z},
        .translation = {translation.x, translation.y, translation.z},
    };
}

void expect_matrix_near(
    const float4x4 &actual, const float4x4 &expected,
    float tolerance, luisa::string_view label) {
    for (auto column = 0u; column < 4u; column++) {
        for (auto row = 0u; row < 4u; row++) {
            expect(std::abs(
                       actual[column][row] -
                       expected[column][row]) <= tolerance)
                << luisa::format(
                       "{}[{}][{}] mismatch", label, column, row);
        }
    }
}

void expect_srt_near(
    const MotionInstanceTransformSRT &actual,
    const MotionInstanceTransformSRT &expected,
    float tolerance, luisa::string_view label) {
    auto check = [&](const float *lhs, const float *rhs,
                     size_t count, luisa::string_view field) {
        for (auto i = size_t{0u}; i < count; i++) {
            expect(std::abs(lhs[i] - rhs[i]) <= tolerance)
                << luisa::format(
                       "{}.{}[{}] mismatch", label, field, i);
        }
    };
    check(actual.pivot, expected.pivot, 3u, "pivot");
    check(actual.quaternion, expected.quaternion, 4u, "quaternion");
    check(actual.scale, expected.scale, 3u, "scale");
    check(actual.shear, expected.shear, 3u, "shear");
    check(actual.translation, expected.translation, 3u, "translation");
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    Context context{argc > 0 ? argv[0] : ""};

    std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    std::array motion_vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(-1.0f, -1.0f, -2.0f),
        make_float3(1.0f, -1.0f, -2.0f),
        make_float3(0.0f, 1.0f, -2.0f)};
    std::array triangles{Triangle{0u, 1u, 2u}};

    for (auto width : std::array{1u, 2u, 4u, 8u, 16u}) {
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(width);
        auto device = context.create_device("simd", &config);
        auto stream = device.create_stream();

        auto vertex_buffer =
            device.create_buffer<float3>(vertices.size());
        auto triangle_buffer =
            device.create_buffer<Triangle>(triangles.size());
        auto mesh = device.create_mesh(
            vertex_buffer, triangle_buffer);
        auto motion_vertex_buffer =
            device.create_buffer<float3>(motion_vertices.size());
        AccelOption motion_mesh_option{};
        motion_mesh_option.motion.keyframe_count = 2u;
        motion_mesh_option.motion.time_start = 0.0f;
        motion_mesh_option.motion.time_end = 1.0f;
        auto motion_mesh = device.create_mesh(
            motion_vertex_buffer, triangle_buffer,
            motion_mesh_option);
        AccelMotionOption matrix_instance_option{};
        matrix_instance_option.keyframe_count = 2u;
        matrix_instance_option.time_start = 0.0f;
        matrix_instance_option.time_end = 1.0f;
        matrix_instance_option.mode = AccelMotionMode::MATRIX;
        auto matrix_motion_instance = device.create_motion_instance(
            mesh, matrix_instance_option);
        std::array matrix_instance_keys{
            make_float4x4(1.0f),
            translation(make_float3(0.0f, 0.0f, -2.0f))};
        matrix_motion_instance.set_keyframes(
            luisa::span{matrix_instance_keys});
        AccelMotionOption srt_instance_option{};
        srt_instance_option.keyframe_count = 2u;
        srt_instance_option.time_start = 0.0f;
        srt_instance_option.time_end = 1.0f;
        srt_instance_option.mode = AccelMotionMode::SRT;
        auto srt_motion_instance = device.create_motion_instance(
            mesh, srt_instance_option);
        std::array srt_instance_keys{
            make_srt(
                make_float3(0.0f), make_float4(0.0f, 0.0f, 0.0f, 1.0f),
                make_float3(1.0f), make_float3(0.0f),
                make_float3(3.0f, 0.0f, 0.0f)),
            make_srt(
                make_float3(0.0f),
                make_float4(
                    0.0f, 0.0f, 0.7071067811865475f,
                    0.7071067811865475f),
                make_float3(1.0f), make_float3(0.0f),
                make_float3(3.0f, 0.0f, -2.0f))};
        srt_motion_instance.set_keyframes(
            luisa::span{srt_instance_keys});
        // Keep the outer-affine resource isolated from the device-authored
        // SRT mutation tests below. Its host build generation is changed
        // independently to verify that an Accel build imports rebuilt motion
        // keyframes even when the Accel modification list is empty.
        auto outer_srt_motion_instance = device.create_motion_instance(
            mesh, srt_instance_option);
        outer_srt_motion_instance.set_keyframes(
            luisa::span{srt_instance_keys});
        auto accel = device.create_accel({.allow_update = true});
        accel.emplace_back(
            mesh, make_float4x4(1.0f), 0x1u, true, 11u);
        accel.emplace_back(
            mesh, translation(make_float3(0.0f, 0.0f, -1.0f)),
            0x2u, true, 22u);
        auto motion_accel = device.create_accel();
        motion_accel.emplace_back(
            motion_mesh, make_float4x4(1.0f), 0x1u, true, 33u);
        auto instance_motion_accel = device.create_accel({.allow_update = true});
        instance_motion_accel.emplace_back(
            matrix_motion_instance,
            translation(make_float3(-3.0f, 0.0f, 0.0f)),
            0x4u, true, 44u);
        instance_motion_accel.emplace_back(
            srt_motion_instance, make_float4x4(1.0f),
            0x8u, true, 55u);
        auto outer_srt_accel =
            device.create_accel({.allow_update = true});
        auto outer_srt_transform = make_float4x4(
            make_float4(2.0f, 0.25f, 0.0f, 0.0f),
            make_float4(0.5f, 1.5f, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, 1.0f, 0.0f),
            make_float4(1.0f, -2.0f, -2.0f, 1.0f));
        outer_srt_accel.emplace_back(
            outer_srt_motion_instance, outer_srt_transform,
            0x80u, false, 56u);
        auto query_accel = device.create_accel();
        query_accel.emplace_back(
            mesh, make_float4x4(1.0f), 0x10u, false, 66u);
        query_accel.emplace_back(
            mesh, translation(make_float3(0.0f, 0.0f, -2.0f)),
            0x10u, false, 77u);
        auto deep_query_accel = device.create_accel();
        for (auto candidate = 0u;
             candidate < deep_query_candidate_count; candidate++) {
            deep_query_accel.emplace_back(
                mesh,
                translation(make_float3(
                    0.0f, 0.0f,
                    -2.0f * static_cast<float>(candidate))),
                0x20u, false, candidate);
        }
        auto opacity_accel = device.create_accel({.allow_update = true});
        for (auto instance = 0u; instance < thread_count; instance++) {
            auto x =
                (static_cast<float>(instance) -
                 static_cast<float>(thread_count / 2u)) *
                3.0f;
            opacity_accel.emplace_back(
                mesh, translation(make_float3(x, 0.0f, 0.0f)),
                0x40u, false, instance);
        }
        auto buffer_only_accel =
            device.create_accel({.allow_update = true});
        buffer_only_accel.emplace_back(
            mesh, make_float4x4(1.0f), 0x1u, true, 10u);

        std::array<Ray, thread_count> host_rays{};
        std::array<uint, thread_count> host_masks{};
        for (auto i = 0u; i < thread_count; i++) {
            auto mode = i % 7u;
            auto direction = mode == 4u ?
                                 make_float3(0.0f, 0.0f, 1.0f) :
                                 make_float3(0.0f, 0.0f, -1.0f);
            auto t_min = mode == 6u ? 1.5f : 0.0f;
            auto t_max = mode == 5u ? 0.5f : 10.0f;
            host_rays[i] = make_host_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                direction, t_min, t_max);
            host_masks[i] =
                mode == 0u || mode == 5u ? 0x1u :
                mode == 1u || mode == 6u ? 0x2u :
                mode == 3u               ? 0u :
                                           0x3u;
        }

        auto rays = device.create_buffer<Ray>(thread_count);
        auto masks = device.create_buffer<uint>(thread_count);
        auto ids = device.create_buffer<uint4>(thread_count);
        auto details = device.create_buffer<float4>(thread_count);
        auto motion_ids = device.create_buffer<uint4>(thread_count);
        auto motion_details = device.create_buffer<float4>(thread_count);
        auto instance_metadata = device.create_buffer<uint4>(thread_count);
        auto instance_transforms =
            device.create_buffer<float4x4>(thread_count);

        Kernel1D kernel = [width](
                              BufferVar<Ray> ray_buffer,
                              BufferUInt mask_buffer,
                              BufferUInt4 id_buffer,
                              BufferFloat4 detail_buffer,
                              AccelVar scene,
                              AccelVar motion_scene,
                              BufferUInt4 motion_id_buffer,
                              BufferFloat4 motion_detail_buffer,
                              BufferUInt4 instance_metadata_buffer,
                              BufferFloat4x4 instance_transform_buffer) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto ray = ray_buffer.read(index);
            AccelTraceOptions options{
                .visibility_mask = mask_buffer.read(index)};
            auto hit = scene.intersect(ray, options);
            auto any = scene.intersect_any(ray, options);

            auto uniform_ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            auto uniform_hit = scene.intersect(
                uniform_ray,
                AccelTraceOptions{.visibility_mask = 0x1u});
            id_buffer.write(
                index,
                make_uint4(
                    hit->inst, hit->prim,
                    cast<uint>(any), uniform_hit->inst));
            detail_buffer.write(
                index,
                make_float4(
                    hit->bary.x, hit->bary.y,
                    hit->committed_ray_t,
                    uniform_hit->committed_ray_t));

            auto motion_time = cast<float>(index % 5u) * 0.25f;
            auto motion_ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            auto motion_options = AccelTraceOptions{
                .visibility_mask = mask_buffer.read(index)};
            auto motion_hit = motion_scene.intersect_motion(
                motion_ray, motion_time, motion_options);
            auto motion_any = motion_scene.intersect_any_motion(
                motion_ray, motion_time, motion_options);
            auto uniform_motion_hit = motion_scene.intersect_motion(
                motion_ray, 0.5f,
                AccelTraceOptions{.visibility_mask = 0x1u});
            motion_id_buffer.write(
                index,
                make_uint4(
                    motion_hit->inst, motion_hit->prim,
                    cast<uint>(motion_any),
                    uniform_motion_hit->inst));
            motion_detail_buffer.write(
                index,
                make_float4(
                    motion_hit->bary.x, motion_hit->bary.y,
                    motion_hit->committed_ray_t,
                    uniform_motion_hit->committed_ray_t));

            auto instance_index = index & 1u;
            instance_metadata_buffer.write(
                index,
                make_uint4(
                    instance_index,
                    scene.instance_user_id(instance_index),
                    scene.instance_visibility_mask(instance_index),
                    scene.instance_user_id(1u)));
            instance_transform_buffer.write(
                index, scene.instance_transform(instance_index));
        };
        auto shader = device.compile(kernel);

        std::array<uint4, thread_count> host_ids{};
        std::array<float4, thread_count> host_details{};
        std::array<uint4, thread_count> host_motion_ids{};
        std::array<float4, thread_count> host_motion_details{};
        std::array<uint4, thread_count> host_instance_metadata{};
        std::array<float4x4, thread_count> host_instance_transforms{};
        stream << vertex_buffer.copy_from(luisa::span{vertices})
               << motion_vertex_buffer.copy_from(
                      luisa::span{motion_vertices})
               << triangle_buffer.copy_from(luisa::span{triangles})
               << rays.copy_from(luisa::span{host_rays})
               << masks.copy_from(luisa::span{host_masks})
               << mesh.build()
               << motion_mesh.build()
               << matrix_motion_instance.build()
               << srt_motion_instance.build()
               << outer_srt_motion_instance.build()
               << accel.build()
               << motion_accel.build()
               << instance_motion_accel.build()
               << outer_srt_accel.build()
               << query_accel.build()
               << deep_query_accel.build()
               << opacity_accel.build()
               << buffer_only_accel.build()
               << shader(
                      rays, masks, ids, details, accel,
                      motion_accel, motion_ids, motion_details,
                      instance_metadata, instance_transforms)
                      .dispatch(thread_count)
               << ids.copy_to(luisa::span{host_ids})
               << details.copy_to(luisa::span{host_details})
               << motion_ids.copy_to(luisa::span{host_motion_ids})
               << motion_details.copy_to(
                      luisa::span{host_motion_details})
               << instance_metadata.copy_to(
                      luisa::span{host_instance_metadata})
               << instance_transforms.copy_to(
                      luisa::span{host_instance_transforms})
               << synchronize();

        for (auto i = 0u; i < thread_count; i++) {
            auto mode = i % 7u;
            auto expected_instance =
                mode == 0u || mode == 2u ? 0u :
                mode == 1u || mode == 6u ? 1u :
                                           ~0u;
            auto expected_any = expected_instance == ~0u ? 0u : 1u;
            expect(host_ids[i].x == expected_instance)
                << "closest-hit instance mismatch";
            expect(host_ids[i].y ==
                   (expected_instance == ~0u ? ~0u : 0u))
                << "closest-hit primitive mismatch";
            expect(host_ids[i].z == expected_any)
                << "any-hit result mismatch";
            expect(host_ids[i].w == 0u)
                << "uniform closest-hit instance mismatch";
            expect(std::abs(host_details[i].w - 1.0f) <= 1.0e-6f)
                << "uniform closest-hit distance mismatch";
            if (expected_instance != ~0u) {
                auto expected_t = expected_instance == 0u ? 1.0f : 2.0f;
                expect(std::abs(host_details[i].x - 0.25f) <= 1.0e-6f)
                    << "closest-hit barycentric u mismatch";
                expect(std::abs(host_details[i].y - 0.5f) <= 1.0e-6f)
                    << "closest-hit barycentric v mismatch";
                expect(std::abs(host_details[i].z - expected_t) <= 1.0e-6f)
                    << "closest-hit distance mismatch";
            }

            auto motion_hit = (host_masks[i] & 0x1u) != 0u;
            auto expected_motion_instance = motion_hit ? 0u : ~0u;
            expect(host_motion_ids[i].x == expected_motion_instance)
                << "motion closest-hit instance mismatch";
            expect(host_motion_ids[i].y == (motion_hit ? 0u : ~0u))
                << "motion closest-hit primitive mismatch";
            expect(host_motion_ids[i].z == (motion_hit ? 1u : 0u))
                << "motion any-hit result mismatch";
            expect(host_motion_ids[i].w == 0u)
                << "uniform motion closest-hit instance mismatch";
            expect(std::abs(host_motion_details[i].w - 2.0f) <= 1.0e-6f)
                << "uniform motion closest-hit distance mismatch";
            if (motion_hit) {
                auto motion_time = static_cast<float>(i % 5u) * 0.25f;
                auto expected_motion_t = 1.0f + 2.0f * motion_time;
                expect(std::abs(host_motion_details[i].x - 0.25f) <= 1.0e-6f)
                    << "motion closest-hit barycentric u mismatch";
                expect(std::abs(host_motion_details[i].y - 0.5f) <= 1.0e-6f)
                    << "motion closest-hit barycentric v mismatch";
                expect(std::abs(host_motion_details[i].z - expected_motion_t) <=
                       1.0e-6f)
                    << "motion closest-hit distance mismatch";
            }

            auto instance_index = i & 1u;
            auto expected_user_id = instance_index == 0u ? 11u : 22u;
            auto expected_visibility = instance_index == 0u ? 0x1u : 0x2u;
            expect(static_cast<bool>(
                all(host_instance_metadata[i] == make_uint4(
                                                     instance_index,
                                                     expected_user_id,
                                                     expected_visibility,
                                                     22u))))
                << "instance metadata query mismatch";
            auto expected_transform = instance_index == 0u ?
                                          make_float4x4(1.0f) :
                                          translation(make_float3(0.0f, 0.0f, -1.0f));
            for (auto column = 0u; column < 4u; column++) {
                expect(static_cast<bool>(
                    all(host_instance_transforms[i][column] ==
                        expected_transform[column])))
                    << "instance transform query mismatch";
            }
        }

        Kernel1D inspect_buffer_only = [width](
                                           AccelVar scene,
                                           BufferUInt4 metadata,
                                           BufferFloat translations,
                                           BufferUInt hits,
                                           BufferUInt query_hits,
                                           BufferUInt query_any_hits,
                                           UInt metadata_count) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            $if (index < metadata_count) {
                auto transform = scene.instance_transform(index);
                metadata.write(
                    index,
                    make_uint4(
                        index, scene.instance_user_id(index),
                        scene.instance_visibility_mask(index), 0u));
                translations.write(index, transform[3u].x);
            };
            auto x = select(
                select(-3.0f, 3.0f, index == 1u),
                0.0f, index == 0u);
            auto ray = make_ray(
                make_float3(x, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            auto options = AccelTraceOptions{
                .visibility_mask = 1u << index};
            auto hit = scene.intersect(ray, options);
            auto query_hit = scene.traverse(ray, options)
                                 .on_surface_candidate(
                                     [](SurfaceCandidate &candidate) noexcept {
                                         candidate.commit();
                                     })
                                 .on_procedural_candidate(
                                     [](ProceduralCandidate &) noexcept {})
                                 .trace();
            auto query_any_hit = scene.traverse_any(ray, options)
                                     .on_surface_candidate(
                                         [](SurfaceCandidate &candidate) noexcept {
                                             candidate.commit();
                                         })
                                     .on_procedural_candidate(
                                         [](ProceduralCandidate &) noexcept {})
                                     .trace();
            hits.write(index, hit->inst);
            query_hits.write(index, query_hit->inst);
            query_any_hits.write(index, query_any_hit->inst);
        };
        auto inspect_buffer_only_shader =
            device.compile(inspect_buffer_only);
        auto buffer_only_metadata = device.create_buffer<uint4>(2u);
        auto buffer_only_translations = device.create_buffer<float>(2u);
        auto buffer_only_hits = device.create_buffer<uint>(3u);
        auto buffer_only_query_hits = device.create_buffer<uint>(3u);
        auto buffer_only_query_any_hits = device.create_buffer<uint>(3u);
        std::array<uint4, 2u> host_buffer_only_metadata{};
        std::array<float, 2u> host_buffer_only_translations{};
        std::array<uint, 3u> host_buffer_only_hits{};
        std::array<uint, 3u> host_buffer_only_query_hits{};
        std::array<uint, 3u> host_buffer_only_query_any_hits{};

        buffer_only_accel.set_transform_on_update(
            0u, translation(make_float3(3.0f, 0.0f, 0.0f)));
        buffer_only_accel.set_visibility_on_update(0u, 0x2u);
        buffer_only_accel.set_instance_user_id_on_update(0u, 20u);
        buffer_only_accel.emplace_back(
            mesh, translation(make_float3(-3.0f, 0.0f, 0.0f)),
            0x4u, true, 30u);
        stream << buffer_only_accel.update_instance_buffer()
               << inspect_buffer_only_shader(
                      buffer_only_accel,
                      buffer_only_metadata,
                      buffer_only_translations,
                      buffer_only_hits,
                      buffer_only_query_hits,
                      buffer_only_query_any_hits, 2u)
                      .dispatch(3u)
               << buffer_only_metadata.copy_to(
                      luisa::span{host_buffer_only_metadata})
               << buffer_only_translations.copy_to(
                      luisa::span{host_buffer_only_translations})
               << buffer_only_hits.copy_to(
                      luisa::span{host_buffer_only_hits})
               << buffer_only_query_hits.copy_to(
                      luisa::span{host_buffer_only_query_hits})
               << buffer_only_query_any_hits.copy_to(
                      luisa::span{host_buffer_only_query_any_hits})
               << synchronize();
        expect(static_cast<bool>(
            all(host_buffer_only_metadata[0u] ==
                make_uint4(0u, 20u, 0x2u, 0u))))
            << "buffer-only existing metadata mismatch";
        expect(static_cast<bool>(
            all(host_buffer_only_metadata[1u] ==
                make_uint4(1u, 30u, 0x4u, 0u))))
            << "buffer-only appended metadata mismatch";
        expect(std::abs(host_buffer_only_translations[0u] - 3.0f) <=
               1.0e-6f)
            << "buffer-only existing transform mismatch";
        expect(std::abs(host_buffer_only_translations[1u] + 3.0f) <=
               1.0e-6f)
            << "buffer-only appended transform mismatch";
        expect(host_buffer_only_hits ==
               std::array<uint, 3u>{0u, ~0u, ~0u})
            << "buffer-only update must not mutate the committed BVH";
        expect(host_buffer_only_query_hits == host_buffer_only_hits)
            << "buffer-only query view diverged from the committed BVH";
        expect(host_buffer_only_query_any_hits == host_buffer_only_hits)
            << "buffer-only any-query view diverged from the committed BVH";

        host_buffer_only_hits.fill(0u);
        host_buffer_only_query_hits.fill(0u);
        host_buffer_only_query_any_hits.fill(0u);
        stream << buffer_only_accel.build(
                      Accel::BuildRequest::PREFER_UPDATE)
               << inspect_buffer_only_shader(
                      buffer_only_accel,
                      buffer_only_metadata,
                      buffer_only_translations,
                      buffer_only_hits,
                      buffer_only_query_hits,
                      buffer_only_query_any_hits, 2u)
                      .dispatch(3u)
               << buffer_only_hits.copy_to(
                      luisa::span{host_buffer_only_hits})
               << buffer_only_query_hits.copy_to(
                      luisa::span{host_buffer_only_query_hits})
               << buffer_only_query_any_hits.copy_to(
                      luisa::span{host_buffer_only_query_any_hits})
               << synchronize();
        expect(host_buffer_only_hits ==
               std::array<uint, 3u>{~0u, 0u, 1u})
            << "ordinary build did not commit deferred buffer-only updates";
        expect(host_buffer_only_query_hits == host_buffer_only_hits)
            << "deferred-build query view mismatch";
        expect(host_buffer_only_query_any_hits == host_buffer_only_hits)
            << "deferred-build any-query view mismatch";

        buffer_only_accel.pop_back();
        host_buffer_only_hits.fill(0u);
        host_buffer_only_query_hits.fill(0u);
        host_buffer_only_query_any_hits.fill(0u);
        stream << buffer_only_accel.update_instance_buffer()
               << inspect_buffer_only_shader(
                      buffer_only_accel,
                      buffer_only_metadata,
                      buffer_only_translations,
                      buffer_only_hits,
                      buffer_only_query_hits,
                      buffer_only_query_any_hits, 1u)
                      .dispatch(3u)
               << buffer_only_hits.copy_to(
                      luisa::span{host_buffer_only_hits})
               << buffer_only_query_hits.copy_to(
                      luisa::span{host_buffer_only_query_hits})
               << buffer_only_query_any_hits.copy_to(
                      luisa::span{host_buffer_only_query_any_hits})
               << synchronize();
        expect(host_buffer_only_hits ==
               std::array<uint, 3u>{~0u, 0u, 1u})
            << "buffer-only shrink must retain the committed BVH view";
        expect(host_buffer_only_query_hits == host_buffer_only_hits)
            << "buffer-only shrink lost committed query metadata";
        expect(host_buffer_only_query_any_hits == host_buffer_only_hits)
            << "buffer-only shrink lost committed any-query metadata";

        host_buffer_only_hits.fill(0u);
        host_buffer_only_query_hits.fill(0u);
        host_buffer_only_query_any_hits.fill(0u);
        stream << buffer_only_accel.build(
                      Accel::BuildRequest::PREFER_UPDATE)
               << inspect_buffer_only_shader(
                      buffer_only_accel,
                      buffer_only_metadata,
                      buffer_only_translations,
                      buffer_only_hits,
                      buffer_only_query_hits,
                      buffer_only_query_any_hits, 1u)
                      .dispatch(3u)
               << buffer_only_hits.copy_to(
                      luisa::span{host_buffer_only_hits})
               << buffer_only_query_hits.copy_to(
                      luisa::span{host_buffer_only_query_hits})
               << buffer_only_query_any_hits.copy_to(
                      luisa::span{host_buffer_only_query_any_hits})
               << synchronize();
        expect(host_buffer_only_hits ==
               std::array<uint, 3u>{~0u, 0u, ~0u})
            << "ordinary build did not commit a deferred shrink";
        expect(host_buffer_only_query_hits == host_buffer_only_hits)
            << "deferred-shrink query view mismatch";
        expect(host_buffer_only_query_any_hits == host_buffer_only_hits)
            << "deferred-shrink any-query view mismatch";

        Kernel1D trace_query = [width](
                                   AccelVar scene,
                                   BufferUInt4 metadata,
                                   BufferFloat2 details,
                                   BufferUInt4 any_metadata,
                                   BufferFloat2 any_details,
                                   BufferUInt4 terminate_metadata) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto target = index & 1u;
            auto ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            UInt callback_count = 0u;
            Float callback_tmax = -1.0f;
            auto committed = scene.traverse(
                                      ray,
                                      AccelTraceOptions{
                                          .visibility_mask = 0x10u})
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         auto hit = candidate.hit();
                                         callback_count += 1u;
                                         $if (hit->inst == target) {
                                             candidate.commit();
                                             callback_tmax =
                                                 candidate.ray()->t_max();
                                         };
                                     })
                                 .on_procedural_candidate(
                                     [](ProceduralCandidate &) noexcept {})
                                 .trace();
            metadata.write(
                index,
                make_uint4(
                    committed->hit_type, committed->inst,
                    committed->prim, callback_count));
            details.write(
                index,
                make_float2(
                    committed->committed_ray_t, callback_tmax));

            UInt any_callback_count = 0u;
            Float any_callback_tmax = -1.0f;
            auto any_committed = scene.traverse_any(
                                          ray,
                                          AccelTraceOptions{
                                              .visibility_mask = 0x10u})
                                     .on_surface_candidate(
                                         [&](SurfaceCandidate &candidate) noexcept {
                                             auto hit = candidate.hit();
                                             any_callback_count += 1u;
                                             $if (hit->inst == target) {
                                                 candidate.commit();
                                                 any_callback_tmax =
                                                     candidate.ray()->t_max();
                                             };
                                         })
                                     .on_procedural_candidate(
                                         [](ProceduralCandidate &) noexcept {})
                                     .trace();
            any_metadata.write(
                index,
                make_uint4(
                    any_committed->hit_type, any_committed->inst,
                    any_committed->prim, any_callback_count));
            any_details.write(
                index,
                make_float2(
                    any_committed->committed_ray_t,
                    any_callback_tmax));

            UInt terminate_callback_count = 0u;
            auto terminated = scene.traverse(
                                       ray,
                                       AccelTraceOptions{
                                           .visibility_mask = 0x10u})
                                  .on_surface_candidate(
                                      [&](SurfaceCandidate &candidate) noexcept {
                                          terminate_callback_count += 1u;
                                          $if ((index & 1u) == 0u) {
                                              candidate.terminate();
                                          }
                                          $else {
                                              candidate.commit();
                                          };
                                      })
                                  .on_procedural_candidate(
                                      [](ProceduralCandidate &) noexcept {})
                                  .trace();
            terminate_metadata.write(
                index,
                make_uint4(
                    terminated->hit_type, terminated->inst,
                    terminated->prim, terminate_callback_count));
        };
        auto trace_query_shader = device.compile(trace_query);
        auto query_metadata = device.create_buffer<uint4>(5u);
        auto query_details = device.create_buffer<float2>(5u);
        auto query_any_metadata = device.create_buffer<uint4>(5u);
        auto query_any_details = device.create_buffer<float2>(5u);
        auto query_terminate_metadata = device.create_buffer<uint4>(5u);
        std::array<uint4, 5u> host_query_metadata{};
        std::array<float2, 5u> host_query_details{};
        std::array<uint4, 5u> host_query_any_metadata{};
        std::array<float2, 5u> host_query_any_details{};
        std::array<uint4, 5u> host_query_terminate_metadata{};
        stream << trace_query_shader(
                      query_accel, query_metadata, query_details,
                      query_any_metadata, query_any_details,
                      query_terminate_metadata)
                      .dispatch(5u)
               << query_metadata.copy_to(luisa::span{host_query_metadata})
               << query_details.copy_to(luisa::span{host_query_details})
               << query_any_metadata.copy_to(
                      luisa::span{host_query_any_metadata})
               << query_any_details.copy_to(
                      luisa::span{host_query_any_details})
               << query_terminate_metadata.copy_to(
                      luisa::span{host_query_terminate_metadata})
               << synchronize();
        for (auto i = 0u; i < host_query_metadata.size(); i++) {
            auto target = i & 1u;
            expect(static_cast<bool>(
                all(host_query_metadata[i] == make_uint4(
                                                  static_cast<uint32_t>(
                                                      HitType::Surface),
                                                  target, 0u,
                                                  target + 1u))))
                << "triangle ray-query metadata mismatch";
            auto expected_distance = target == 0u ? 1.0f : 3.0f;
            expect(std::abs(
                       host_query_details[i].x - expected_distance) <=
                   1.0e-5f)
                << "triangle ray-query distance mismatch";
            expect(std::abs(
                       host_query_details[i].y - expected_distance) <=
                   1.0e-5f)
                << "triangle ray-query committed tmax mismatch";
            expect(static_cast<bool>(
                all(host_query_any_metadata[i] == make_uint4(
                                                      static_cast<uint32_t>(
                                                          HitType::Surface),
                                                      target, 0u,
                                                      target + 1u))))
                << "triangle any ray-query metadata mismatch";
            expect(std::abs(
                       host_query_any_details[i].x - expected_distance) <=
                   1.0e-5f)
                << "triangle any ray-query distance mismatch";
            expect(std::abs(
                       host_query_any_details[i].y - expected_distance) <=
                   1.0e-5f)
                << "triangle any ray-query committed tmax mismatch";
            auto expected_terminated = (i & 1u) == 0u ?
                                           make_uint4(
                                               static_cast<uint32_t>(
                                                   HitType::Miss),
                                               ~0u, ~0u, 1u) :
                                           make_uint4(
                                               static_cast<uint32_t>(
                                                   HitType::Surface),
                                               0u, 0u, 1u);
            expect(static_cast<bool>(
                all(host_query_terminate_metadata[i] ==
                    expected_terminated)))
                << "triangle ray-query terminate mismatch";
        }

        // Capture-free, surface-only handlers are eligible for the Embree
        // in-filter packet pipeline. Even lanes reject the nearer instance
        // and commit the farther one; odd lanes miss, forming sparse callback
        // masks. Thirty-five lanes exercise every width's inactive tail.
        Kernel1D trace_surface_filter_pipeline = [width](
                                                     AccelVar scene,
                                                     BufferUInt4 metadata,
                                                     BufferFloat2 distances) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto direction = select(
                make_float3(0.0f, 0.0f, -1.0f),
                make_float3(0.0f, 0.0f, 1.0f),
                (index & 1u) != 0u);
            auto ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                direction, 0.0f, 10.0f);
            auto options = AccelTraceOptions{
                .visibility_mask = 0x10u};
            auto closest = scene.traverse(ray, options)
                               .on_surface_candidate(
                                   [](SurfaceCandidate &candidate) noexcept {
                                       $if (candidate.hit()->inst == 1u) {
                                           candidate.commit();
                                       };
                                   })
                               .on_procedural_candidate(
                                   [](ProceduralCandidate &) noexcept {})
                               .trace();
            auto any = scene.traverse_any(ray, options)
                           .on_surface_candidate(
                               [](SurfaceCandidate &candidate) noexcept {
                                   $if (candidate.hit()->inst == 1u) {
                                       candidate.commit();
                                   };
                               })
                           .on_procedural_candidate(
                               [](ProceduralCandidate &) noexcept {})
                           .trace();
            metadata.write(
                index,
                make_uint4(
                    closest->hit_type, closest->inst,
                    any->hit_type, any->inst));
            distances.write(
                index,
                make_float2(
                    closest->committed_ray_t,
                    any->committed_ray_t));
        };
        auto trace_surface_filter_pipeline_shader =
            device.compile(trace_surface_filter_pipeline);
        auto surface_filter_metadata =
            device.create_buffer<uint4>(thread_count);
        auto surface_filter_distances =
            device.create_buffer<float2>(thread_count);
        std::array<uint4, thread_count>
            host_surface_filter_metadata{};
        std::array<float2, thread_count>
            host_surface_filter_distances{};
        stream << trace_surface_filter_pipeline_shader(
                      query_accel, surface_filter_metadata,
                      surface_filter_distances)
                      .dispatch(thread_count)
               << surface_filter_metadata.copy_to(
                      luisa::span{host_surface_filter_metadata})
               << surface_filter_distances.copy_to(
                      luisa::span{host_surface_filter_distances})
               << synchronize();
        for (auto i = 0u; i < thread_count; i++) {
            if ((i & 1u) == 0u) {
                expect(static_cast<bool>(
                    all(host_surface_filter_metadata[i] ==
                        make_uint4(
                            static_cast<uint32_t>(HitType::Surface),
                            1u,
                            static_cast<uint32_t>(HitType::Surface),
                            1u))))
                    << luisa::format(
                           "surface-filter hit mismatch at W{} lane {}",
                           width, i);
                expect(std::abs(
                           host_surface_filter_distances[i].x - 3.0f) <=
                           1.0e-5f &&
                       std::abs(
                           host_surface_filter_distances[i].y - 3.0f) <=
                           1.0e-5f)
                    << luisa::format(
                           "surface-filter distance mismatch at W{} lane {}",
                           width, i);
            } else {
                expect(static_cast<bool>(
                    all(host_surface_filter_metadata[i] ==
                        make_uint4(
                            static_cast<uint32_t>(HitType::Miss),
                            ~0u,
                            static_cast<uint32_t>(HitType::Miss),
                            ~0u))))
                    << luisa::format(
                           "surface-filter miss mismatch at W{} lane {}",
                           width, i);
            }
        }

        // Cross the fixed thirty-two-candidate speculative batch boundary. The
        // thirty-third candidate must trigger a continuation scan without
        // duplicating or skipping handler invocations.
        Kernel1D trace_deep_query = [width](
                                        AccelVar scene,
                                        BufferUInt4 metadata) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto ray = make_ray(
                make_float3(0.0f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 128.0f);
            UInt callback_count = 0u;
            auto committed = scene.traverse(
                                      ray,
                                      AccelTraceOptions{
                                          .visibility_mask = 0x20u})
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         callback_count += 1u;
                                         $if (candidate.hit()->inst ==
                                              deep_query_candidate_count - 1u) {
                                             candidate.commit();
                                         };
                                     })
                                 .on_procedural_candidate(
                                     [](ProceduralCandidate &) noexcept {})
                                 .trace();
            metadata.write(
                index,
                make_uint4(
                    committed->hit_type, committed->inst,
                    committed->prim, callback_count));
        };
        auto trace_deep_query_shader = device.compile(trace_deep_query);
        auto deep_query_metadata = device.create_buffer<uint4>(5u);
        std::array<uint4, 5u> host_deep_query_metadata{};
        stream << trace_deep_query_shader(
                      deep_query_accel, deep_query_metadata)
                      .dispatch(5u)
               << deep_query_metadata.copy_to(
                      luisa::span{host_deep_query_metadata})
               << synchronize();
        for (auto metadata : host_deep_query_metadata) {
            expect(static_cast<bool>(
                all(metadata == make_uint4(
                                    static_cast<uint32_t>(
                                        HitType::Surface),
                                    deep_query_candidate_count - 1u,
                                    0u, deep_query_candidate_count))))
                << "deep triangle ray-query continuation mismatch";
        }

        Kernel1D trace_motion_instances = [width](
                                              AccelVar scene,
                                              BufferUInt2 ids,
                                              BufferFloat distances) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto instance = index & 1u;
            auto time = cast<float>(index >> 1u);
            auto origin_x = select(-3.0f, 3.0f, instance != 0u);
            auto ray = make_ray(
                make_float3(origin_x, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            auto hit = scene.intersect_motion(
                ray, time,
                AccelTraceOptions{
                    .visibility_mask = 0x4u << instance});
            auto any = scene.intersect_any_motion(
                ray, time,
                AccelTraceOptions{
                    .visibility_mask = 0x4u << instance});
            ids.write(index, make_uint2(hit->inst, cast<uint>(any)));
            distances.write(index, hit->committed_ray_t);
        };
        auto trace_motion_instances_shader =
            device.compile(trace_motion_instances);
        auto motion_instance_ids = device.create_buffer<uint2>(4u);
        auto motion_instance_distances = device.create_buffer<float>(4u);
        std::array<uint2, 4u> host_motion_instance_ids{};
        std::array<float, 4u> host_motion_instance_distances{};
        stream << trace_motion_instances_shader(
                      instance_motion_accel,
                      motion_instance_ids,
                      motion_instance_distances)
                      .dispatch(4u)
               << motion_instance_ids.copy_to(
                      luisa::span{host_motion_instance_ids})
               << motion_instance_distances.copy_to(
                      luisa::span{host_motion_instance_distances})
               << synchronize();
        for (auto i = 0u; i < 4u; i++) {
            auto expected_instance = i & 1u;
            auto expected_distance = i < 2u ? 1.0f : 3.0f;
            expect(static_cast<bool>(
                all(host_motion_instance_ids[i] ==
                    make_uint2(expected_instance, 1u))))
                << "motion-instance packet traversal id mismatch";
            expect(std::abs(
                       host_motion_instance_distances[i] -
                       expected_distance) <= 1.0e-5f)
                << "motion-instance packet traversal distance mismatch";
        }

        Kernel1D trace_motion_query = [width](
                                          AccelVar scene,
                                          BufferUInt4 metadata,
                                          BufferFloat2 distances,
                                          BufferUInt2 filter_metadata,
                                          BufferFloat2 filter_distances) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto instance = index & 1u;
            auto time = cast<float>(index >> 1u);
            auto origin_x = select(-3.0f, 3.0f, instance != 0u);
            auto ray = make_ray(
                make_float3(origin_x, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            auto options = AccelTraceOptions{
                .visibility_mask = 0x4u << instance};
            UInt all_callback_count = 0u;
            auto all = scene.traverse_motion(ray, time, options)
                           .on_surface_candidate(
                               [&](SurfaceCandidate &candidate) noexcept {
                                   all_callback_count += 1u;
                                   candidate.commit();
                               })
                           .on_procedural_candidate(
                               [](ProceduralCandidate &) noexcept {})
                           .trace();
            UInt any_callback_count = 0u;
            auto any = scene.traverse_any_motion(ray, time, options)
                           .on_surface_candidate(
                               [&](SurfaceCandidate &candidate) noexcept {
                                   any_callback_count += 1u;
                                   candidate.commit();
                               })
                           .on_procedural_candidate(
                               [](ProceduralCandidate &) noexcept {})
                           .trace();
            auto filter_all = scene.traverse_motion(ray, time, options)
                                  .on_surface_candidate(
                                      [](SurfaceCandidate &candidate) noexcept {
                                          candidate.commit();
                                      })
                                  .on_procedural_candidate(
                                      [](ProceduralCandidate &) noexcept {})
                                  .trace();
            auto filter_any = scene.traverse_any_motion(ray, time, options)
                                  .on_surface_candidate(
                                      [](SurfaceCandidate &candidate) noexcept {
                                          candidate.commit();
                                      })
                                  .on_procedural_candidate(
                                      [](ProceduralCandidate &) noexcept {})
                                  .trace();
            metadata.write(
                index,
                make_uint4(
                    all->inst, any->inst,
                    all_callback_count, any_callback_count));
            distances.write(
                index,
                make_float2(
                    all->committed_ray_t,
                    any->committed_ray_t));
            filter_metadata.write(
                index, make_uint2(filter_all->inst, filter_any->inst));
            filter_distances.write(
                index,
                make_float2(
                    filter_all->committed_ray_t,
                    filter_any->committed_ray_t));
        };
        auto trace_motion_query_shader =
            device.compile(trace_motion_query);
        auto motion_query_metadata = device.create_buffer<uint4>(4u);
        auto motion_query_distances = device.create_buffer<float2>(4u);
        auto motion_filter_metadata = device.create_buffer<uint2>(4u);
        auto motion_filter_distances = device.create_buffer<float2>(4u);
        std::array<uint4, 4u> host_motion_query_metadata{};
        std::array<float2, 4u> host_motion_query_distances{};
        std::array<uint2, 4u> host_motion_filter_metadata{};
        std::array<float2, 4u> host_motion_filter_distances{};
        stream << trace_motion_query_shader(
                      instance_motion_accel,
                      motion_query_metadata,
                      motion_query_distances,
                      motion_filter_metadata,
                      motion_filter_distances)
                      .dispatch(4u)
               << motion_query_metadata.copy_to(
                      luisa::span{host_motion_query_metadata})
               << motion_query_distances.copy_to(
                      luisa::span{host_motion_query_distances})
               << motion_filter_metadata.copy_to(
                      luisa::span{host_motion_filter_metadata})
               << motion_filter_distances.copy_to(
                      luisa::span{host_motion_filter_distances})
               << synchronize();
        for (auto i = 0u; i < 4u; i++) {
            auto expected_instance = i & 1u;
            auto expected_distance = i < 2u ? 1.0f : 3.0f;
            expect(static_cast<bool>(
                all(host_motion_query_metadata[i] ==
                    make_uint4(
                        expected_instance, expected_instance,
                        0u, 0u))))
                << "opaque motion ray-query metadata mismatch";
            expect(
                std::abs(
                    host_motion_query_distances[i].x -
                    expected_distance) <= 1.0e-5f &&
                std::abs(
                    host_motion_query_distances[i].y -
                    expected_distance) <= 1.0e-5f)
                << "opaque motion ray-query distance mismatch";
            expect(static_cast<bool>(
                all(host_motion_filter_metadata[i] ==
                    make_uint2(expected_instance, expected_instance))))
                << "surface-filter motion ray-query metadata mismatch";
            expect(
                std::abs(
                    host_motion_filter_distances[i].x -
                    expected_distance) <= 1.0e-5f &&
                std::abs(
                    host_motion_filter_distances[i].y -
                    expected_distance) <= 1.0e-5f)
                << "surface-filter motion ray-query distance mismatch";
        }

        Kernel1D trace_srt_interpolation = [width](
                                               AccelVar scene,
                                               BufferUInt2 ids,
                                               BufferFloat distances) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto ray = make_ray(
                make_float3(3.7071067811865475f, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            auto hit = scene.intersect_motion(
                ray, 0.5f,
                AccelTraceOptions{.visibility_mask = 0x8u});
            auto any = scene.intersect_any_motion(
                ray, 0.5f,
                AccelTraceOptions{.visibility_mask = 0x8u});
            ids.write(0u, make_uint2(hit->inst, cast<uint>(any)));
            distances.write(0u, hit->committed_ray_t);
        };
        auto trace_srt_interpolation_shader =
            device.compile(trace_srt_interpolation);
        stream << trace_srt_interpolation_shader(
                      instance_motion_accel,
                      motion_instance_ids,
                      motion_instance_distances)
                      .dispatch(1u)
               << motion_instance_ids.copy_to(
                      luisa::span{host_motion_instance_ids}.subspan(0u, 1u))
               << motion_instance_distances.copy_to(
                      luisa::span{host_motion_instance_distances}.subspan(
                          0u, 1u))
               << synchronize();
        expect(static_cast<bool>(
            all(host_motion_instance_ids[0u] == make_uint2(1u, 1u))))
            << "SRT quaternion interpolation traversal id mismatch";
        expect(std::abs(host_motion_instance_distances[0u] - 2.0f) <=
               1.0e-5f)
            << "SRT quaternion interpolation traversal distance mismatch";

        // A static outer affine and quaternion-SRT motion form two logical
        // transform levels. The deployed Embree build exposes only one
        // instance-ID stack level, so the SIMD runtime must preserve spherical
        // interpolation through packet forwarding instead of converting the
        // composed endpoints to linearly interpolated matrices. The chosen
        // nonuniform/sheared outer transform does not commute with the
        // forty-five-degree midpoint rotation and therefore catches that
        // approximation. Dispatching thirty-five lanes also exercises every
        // width's inactive tail.
        Kernel1D trace_outer_srt = [width](
                                       AccelVar scene,
                                       BufferFloat3 origins,
                                       BufferUInt4 metadata,
                                       BufferFloat4 details) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto ray = make_ray(
                origins.read(index),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            auto options = AccelTraceOptions{
                .visibility_mask = 0x80u};
            auto hit = scene.intersect_motion(ray, 0.5f, options);
            auto any = scene.intersect_any_motion(ray, 0.5f, options);
            UInt callback_count = 0u;
            auto query = scene.traverse_motion(ray, 0.5f, options)
                             .on_surface_candidate(
                                 [&](SurfaceCandidate &candidate) noexcept {
                                     callback_count += 1u;
                                     candidate.commit();
                                 })
                             .on_procedural_candidate(
                                 [](ProceduralCandidate &) noexcept {})
                             .trace();
            metadata.write(
                index,
                make_uint4(
                    hit->inst, cast<uint>(any),
                    query->inst, callback_count));
            details.write(
                index,
                make_float4(
                    hit->bary.x, hit->bary.y,
                    hit->committed_ray_t,
                    query->committed_ray_t));
        };
        auto trace_outer_srt_shader = device.compile(trace_outer_srt);
        auto outer_srt_origins =
            device.create_buffer<float3>(thread_count);
        auto outer_srt_metadata =
            device.create_buffer<uint4>(thread_count);
        auto outer_srt_details =
            device.create_buffer<float4>(thread_count);
        std::array<uint4, thread_count> host_outer_srt_metadata{};
        std::array<float4, thread_count> host_outer_srt_details{};
        auto check_outer_srt = [&](float3 origin, float expected_t,
                                   luisa::string_view phase) {
            std::array<float3, thread_count> host_origins{};
            host_origins.fill(origin);
            stream << outer_srt_origins.copy_from(
                          luisa::span{host_origins})
                   << trace_outer_srt_shader(
                          outer_srt_accel, outer_srt_origins,
                          outer_srt_metadata, outer_srt_details)
                          .dispatch(thread_count)
                   << outer_srt_metadata.copy_to(
                          luisa::span{host_outer_srt_metadata})
                   << outer_srt_details.copy_to(
                          luisa::span{host_outer_srt_details})
                   << synchronize();
            for (auto i = 0u; i < thread_count; i++) {
                expect(static_cast<bool>(
                    all(host_outer_srt_metadata[i] ==
                        make_uint4(0u, 1u, 0u, 1u))))
                    << luisa::format(
                           "{} metadata mismatch", phase);
                expect(
                    std::abs(host_outer_srt_details[i].x - 0.375f) <=
                        2.0e-5f &&
                    std::abs(host_outer_srt_details[i].y - 0.5f) <=
                        2.0e-5f &&
                    std::abs(
                        host_outer_srt_details[i].z - expected_t) <=
                        2.0e-5f &&
                    std::abs(
                        host_outer_srt_details[i].w - expected_t) <=
                        2.0e-5f)
                    << luisa::format(
                           "{} detail mismatch", phase);
            }
        };
        auto initial_outer_origin = make_float3(
            7.441941738241592f,
            -0.9406407832308854f, 1.0f);
        check_outer_srt(
            initial_outer_origin, 4.0f,
            "initial outer-affine SRT traversal");

        // Rebuilding the MotionInstance is not an Accel modification. The
        // following empty Accel build must still import the new generation.
        auto rebuilt_outer_srt_keys = srt_instance_keys;
        rebuilt_outer_srt_keys[0u].translation[0u] += 1.0f;
        rebuilt_outer_srt_keys[1u].translation[0u] += 1.0f;
        outer_srt_motion_instance.set_keyframes(
            luisa::span{rebuilt_outer_srt_keys});
        stream << outer_srt_motion_instance.build()
               << outer_srt_accel.build(
                      Accel::BuildRequest::PREFER_UPDATE)
               << synchronize();
        auto rebuilt_outer_origin = make_float3(
            9.441941738241592f,
            -0.6906407832308854f, 1.0f);
        check_outer_srt(
            rebuilt_outer_origin, 4.0f,
            "rebuilt outer-affine SRT traversal");

        // A buffer-only transform update changes the public instance table but
        // deliberately keeps the old Embree BVH and its user-geometry payload
        // alive. Traversal therefore stays on the old route until the next
        // ordinary build, which then switches USER -> INSTANCE exactly once.
        outer_srt_accel.set_transform_on_update(
            0u, make_float4x4(1.0f));
        stream << outer_srt_accel.update_instance_buffer()
               << synchronize();
        check_outer_srt(
            rebuilt_outer_origin, 4.0f,
            "buffer-only committed outer-affine SRT traversal");
        stream << outer_srt_accel.build(
                      Accel::BuildRequest::PREFER_UPDATE)
               << synchronize();
        auto identity_outer_origin = make_float3(
            4.176776695296637f,
            0.17677669529663687f, 1.0f);
        check_outer_srt(
            identity_outer_origin, 2.0f,
            "identity outer SRT traversal");

        // Exercise the reverse INSTANCE -> USER route transition and define
        // the singular-composition boundary as a miss. The collapsed x axis
        // still places the ray exactly on the conservative bounds plane, so a
        // callback that is reached must reject before using an invalid inverse.
        auto singular_outer_transform = make_float4x4(
            make_float4(0.0f, 0.0f, 0.0f, 0.0f),
            make_float4(0.0f, 1.0f, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, 1.0f, 0.0f),
            make_float4(
                identity_outer_origin.x, 0.0f, 0.0f, 1.0f));
        outer_srt_accel.set_transform_on_update(
            0u, singular_outer_transform);
        stream << outer_srt_accel.build(
                      Accel::BuildRequest::PREFER_UPDATE)
               << synchronize();
        {
            std::array<float3, thread_count> host_origins{};
            host_origins.fill(identity_outer_origin);
            stream << outer_srt_origins.copy_from(
                          luisa::span{host_origins})
                   << trace_outer_srt_shader(
                          outer_srt_accel, outer_srt_origins,
                          outer_srt_metadata, outer_srt_details)
                          .dispatch(thread_count)
                   << outer_srt_metadata.copy_to(
                          luisa::span{host_outer_srt_metadata})
                   << synchronize();
            for (auto i = 0u; i < thread_count; i++) {
                expect(static_cast<bool>(
                    all(host_outer_srt_metadata[i] ==
                        make_uint4(~0u, 0u, ~0u, 0u))))
                    << "singular outer-affine SRT must miss";
            }
        }

        outer_srt_accel.set_transform_on_update(
            0u, outer_srt_transform);
        stream << outer_srt_accel.build(
                      Accel::BuildRequest::PREFER_UPDATE)
               << synchronize();
        check_outer_srt(
            rebuilt_outer_origin, 4.0f,
            "restored outer-affine SRT traversal");

        Kernel1D read_motion_keyframes = [width](
                                             AccelVar scene,
                                             BufferFloat4x4 matrices,
                                             BufferVar<MotionInstanceTransformSRT> srts) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto key = dispatch_x();
            matrices.write(
                key, scene.instance_motion_matrix(0u, key));
            srts.write(
                key, scene.instance_motion_srt(1u, key));
        };
        Kernel1D write_motion_keyframes = [width](
                                              AccelVar scene,
                                              BufferFloat4x4 matrices,
                                              BufferVar<MotionInstanceTransformSRT> srts) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto key = dispatch_x();
            scene.set_instance_motion_matrix(
                0u, key, matrices.read(key));
            scene.set_instance_motion_srt(
                1u, key, srts.read(key));
        };
        Kernel1D trace_updated_motion_instances = [width](
                                                      AccelVar scene,
                                                      BufferFloat3 origins,
                                                      BufferUInt2 ids,
                                                      BufferFloat distances) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto ray = make_ray(
                origins.read(index),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            auto hit = scene.intersect_motion(
                ray, 0.5f,
                AccelTraceOptions{
                    .visibility_mask = 0x4u << index});
            auto any = scene.intersect_any_motion(
                ray, 0.5f,
                AccelTraceOptions{
                    .visibility_mask = 0x4u << index});
            ids.write(index, make_uint2(hit->inst, cast<uint>(any)));
            distances.write(index, hit->committed_ray_t);
        };
        auto read_motion_keyframes_shader =
            device.compile(read_motion_keyframes);
        auto write_motion_keyframes_shader =
            device.compile(write_motion_keyframes);
        auto trace_updated_motion_instances_shader =
            device.compile(trace_updated_motion_instances);
        auto matrix_keyframe_buffer = device.create_buffer<float4x4>(2u);
        auto srt_keyframe_buffer =
            device.create_buffer<MotionInstanceTransformSRT>(2u);

        auto read_and_check_motion_keyframes =
            [&](luisa::span<const float4x4> expected_matrices,
                luisa::span<const MotionInstanceTransformSRT> expected_srts,
                luisa::string_view phase) {
                std::array<float4x4, 2u> actual_matrices{};
                std::array<MotionInstanceTransformSRT, 2u> actual_srts{};
                stream << read_motion_keyframes_shader(
                              instance_motion_accel,
                              matrix_keyframe_buffer,
                              srt_keyframe_buffer)
                              .dispatch(2u)
                       << matrix_keyframe_buffer.copy_to(
                              luisa::span{actual_matrices})
                       << srt_keyframe_buffer.copy_to(
                              luisa::span{actual_srts})
                       << synchronize();
                for (auto key = 0u; key < 2u; key++) {
                    expect_matrix_near(
                        actual_matrices[key], expected_matrices[key],
                        1.0e-6f,
                        luisa::format("{} matrix key {}", phase, key));
                    expect_srt_near(
                        actual_srts[key], expected_srts[key],
                        1.0e-6f,
                        luisa::format("{} SRT key {}", phase, key));
                }
            };
        read_and_check_motion_keyframes(
            luisa::span{matrix_instance_keys},
            luisa::span{srt_instance_keys},
            "initial device read");

        std::array updated_matrix_keys{
            translation(make_float3(4.0f, 0.0f, 0.0f)),
            translation(make_float3(6.0f, 0.0f, 0.0f))};
        std::array updated_srt_keys{
            make_srt(
                make_float3(0.1f, 0.2f, 0.3f),
                make_float4(0.0f, 0.0f, 0.0f, 1.0f),
                make_float3(1.1f, 1.2f, 1.3f),
                make_float3(0.01f, 0.02f, 0.03f),
                make_float3(0.0f, 1.0f, 0.0f)),
            make_srt(
                make_float3(0.1f, 0.2f, 0.3f),
                make_float4(0.0f, 0.0f, 0.0f, 1.0f),
                make_float3(1.1f, 1.2f, 1.3f),
                make_float3(0.01f, 0.02f, 0.03f),
                make_float3(0.0f, 3.0f, 0.0f))};
        stream << matrix_keyframe_buffer.copy_from(
                      luisa::span{updated_matrix_keys})
               << srt_keyframe_buffer.copy_from(
                      luisa::span{updated_srt_keys})
               << write_motion_keyframes_shader(
                      instance_motion_accel,
                      matrix_keyframe_buffer,
                      srt_keyframe_buffer)
                      .dispatch(2u)
               << instance_motion_accel.build(
                      Accel::BuildRequest::PREFER_UPDATE);
        read_and_check_motion_keyframes(
            luisa::span{updated_matrix_keys},
            luisa::span{updated_srt_keys},
            "post-refit device read");

        std::array updated_motion_origins{
            make_float3(2.0f, 0.0f, 1.0f),
            make_float3(0.1f, 2.2f, 1.3f)};
        auto updated_motion_origin_buffer =
            device.create_buffer<float3>(updated_motion_origins.size());
        auto updated_motion_ids = device.create_buffer<uint2>(2u);
        auto updated_motion_distances = device.create_buffer<float>(2u);
        std::array<uint2, 2u> host_updated_motion_ids{};
        std::array<float, 2u> host_updated_motion_distances{};
        stream << updated_motion_origin_buffer.copy_from(
                      luisa::span{updated_motion_origins})
               << trace_updated_motion_instances_shader(
                      instance_motion_accel,
                      updated_motion_origin_buffer,
                      updated_motion_ids,
                      updated_motion_distances)
                      .dispatch(2u)
               << updated_motion_ids.copy_to(
                      luisa::span{host_updated_motion_ids})
               << updated_motion_distances.copy_to(
                      luisa::span{host_updated_motion_distances})
               << synchronize();
        for (auto i = 0u; i < 2u; i++) {
            expect(static_cast<bool>(
                all(host_updated_motion_ids[i] == make_uint2(i, 1u))))
                << "updated motion-instance traversal id mismatch";
            expect(std::abs(host_updated_motion_distances[i] - 1.0f) <=
                   2.0e-4f)
                << "updated motion-instance traversal distance mismatch";
        }

        std::array updated_transforms{
            translation(make_float3(-3.0f, 0.0f, 0.0f)),
            translation(make_float3(3.0f, 0.0f, 0.0f))};
        std::array updated_ray_origins{
            make_float3(-3.0f, 0.0f, 1.0f),
            make_float3(3.0f, 0.0f, 1.0f)};
        auto updated_transform_buffer =
            device.create_buffer<float4x4>(updated_transforms.size());
        auto updated_origin_buffer =
            device.create_buffer<float3>(updated_ray_origins.size());
        auto updated_metadata = device.create_buffer<uint4>(2u);
        auto updated_details = device.create_buffer<float4>(2u);
        auto updated_transform_output =
            device.create_buffer<float4x4>(2u);

        Kernel1D mutate_instances = [width](
                                        AccelVar scene,
                                        BufferFloat4x4 transforms) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            scene.set_instance_transform(index, transforms.read(index));
            scene.set_instance_visibility(index, 0x4u << index);
            scene.set_instance_user_id(index, 101u + index);
        };
        Kernel1D inspect_updates = [width](
                                       AccelVar scene,
                                       BufferFloat3 origins,
                                       BufferUInt4 metadata,
                                       BufferFloat4 details,
                                       BufferFloat4x4 transforms) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto visibility = 0x4u << index;
            auto ray = make_ray(
                origins.read(index),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            auto options = AccelTraceOptions{
                .visibility_mask = visibility};
            auto hit = scene.intersect(ray, options);
            auto any = scene.intersect_any(ray, options);
            auto transform = scene.instance_transform(index);
            metadata.write(
                index,
                make_uint4(
                    scene.instance_user_id(index),
                    scene.instance_visibility_mask(index),
                    hit->inst, cast<uint>(any)));
            details.write(
                index,
                make_float4(
                    hit->committed_ray_t,
                    transform[3u].x,
                    transform[3u].y,
                    transform[3u].z));
            transforms.write(index, transform);
        };
        auto mutate_shader = device.compile(mutate_instances);
        auto inspect_shader = device.compile(inspect_updates);
        std::array<uint4, 2u> host_updated_metadata{};
        std::array<float4, 2u> host_updated_details{};
        std::array<float4x4, 2u> host_updated_transforms{};
        stream << updated_transform_buffer.copy_from(
                      luisa::span{updated_transforms})
               << updated_origin_buffer.copy_from(
                      luisa::span{updated_ray_origins})
               << mutate_shader(accel, updated_transform_buffer)
                      .dispatch(2u)
               << accel.update_instance_buffer()
               << accel.build(Accel::BuildRequest::PREFER_UPDATE)
               << inspect_shader(
                      accel, updated_origin_buffer,
                      updated_metadata, updated_details,
                      updated_transform_output)
                      .dispatch(2u)
               << updated_metadata.copy_to(
                      luisa::span{host_updated_metadata})
               << updated_details.copy_to(
                      luisa::span{host_updated_details})
               << updated_transform_output.copy_to(
                      luisa::span{host_updated_transforms})
               << synchronize();
        for (auto i = 0u; i < 2u; i++) {
            expect(static_cast<bool>(
                all(host_updated_metadata[i] == make_uint4(
                                                    101u + i,
                                                    0x4u << i,
                                                    i, 1u))))
                << "device instance mutation metadata mismatch";
            expect(std::abs(host_updated_details[i].x - 1.0f) <= 1.0e-6f)
                << "device instance mutation traversal distance mismatch";
            expect(std::abs(
                       host_updated_details[i].y -
                       updated_transforms[i][3u].x) <= 1.0e-6f &&
                   std::abs(
                       host_updated_details[i].z -
                       updated_transforms[i][3u].y) <= 1.0e-6f &&
                   std::abs(
                       host_updated_details[i].w -
                       updated_transforms[i][3u].z) <= 1.0e-6f)
                << "device instance mutation translation mismatch";
            for (auto column = 0u; column < 4u; column++) {
                expect(static_cast<bool>(
                    all(host_updated_transforms[i][column] ==
                        updated_transforms[i][column])))
                    << "device instance mutation transform mismatch";
            }
        }

        Kernel1D mutate_opacity = [width](AccelVar scene) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            scene.set_instance_opaque(index, (index & 1u) == 0u);
        };
        Kernel1D trace_opacity = [width](
                                     AccelVar scene,
                                     BufferUInt4 metadata,
                                     BufferUInt2 empty_metadata) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto index = dispatch_x();
            auto x =
                (cast<float>(index) -
                 static_cast<float>(thread_count / 2u)) *
                3.0f;
            auto ray = make_ray(
                make_float3(x, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f),
                0.0f, 10.0f);
            auto options = AccelTraceOptions{
                .visibility_mask = 0x40u};
            UInt closest_callback_count = 0u;
            auto closest = scene.traverse(ray, options)
                               .on_surface_candidate(
                                   [&](SurfaceCandidate &candidate) noexcept {
                                       closest_callback_count += 1u;
                                       candidate.commit();
                                   })
                               .on_procedural_candidate(
                                   [](ProceduralCandidate &) noexcept {})
                               .trace();
            UInt any_callback_count = 0u;
            auto any = scene.traverse_any(ray, options)
                           .on_surface_candidate(
                               [&](SurfaceCandidate &candidate) noexcept {
                                   any_callback_count += 1u;
                                   candidate.commit();
                               })
                           .on_procedural_candidate(
                               [](ProceduralCandidate &) noexcept {})
                           .trace();
            metadata.write(
                index,
                make_uint4(
                    closest->inst, closest_callback_count,
                    any->inst, any_callback_count));
            // Empty handlers must preserve opacity semantics through the
            // in-filter packet path: opaque triangles auto-commit and
            // non-opaque triangles remain uncommitted.
            auto empty_closest = scene.traverse(ray, options).trace();
            auto empty_any = scene.traverse_any(ray, options).trace();
            empty_metadata.write(
                index,
                make_uint2(
                    empty_closest->inst, empty_any->inst));
        };
        auto mutate_opacity_shader = device.compile(mutate_opacity);
        auto trace_opacity_shader = device.compile(trace_opacity);
        auto opacity_metadata =
            device.create_buffer<uint4>(thread_count);
        auto empty_opacity_metadata =
            device.create_buffer<uint2>(thread_count);
        std::array<uint4, thread_count> host_opacity_metadata{};
        std::array<uint2, thread_count>
            host_empty_opacity_metadata{};
        stream << mutate_opacity_shader(opacity_accel)
                      .dispatch(thread_count)
               << opacity_accel.build(
                      Accel::BuildRequest::PREFER_UPDATE)
               << trace_opacity_shader(
                      opacity_accel, opacity_metadata,
                      empty_opacity_metadata)
                      .dispatch(thread_count)
               << opacity_metadata.copy_to(
                      luisa::span{host_opacity_metadata})
               << empty_opacity_metadata.copy_to(
                      luisa::span{host_empty_opacity_metadata})
               << synchronize();
        for (auto i = 0u; i < thread_count; i++) {
            auto expected_callback_count = i & 1u;
            expect(static_cast<bool>(
                all(host_opacity_metadata[i] ==
                    make_uint4(
                        i, expected_callback_count,
                        i, expected_callback_count))))
                << luisa::format(
                       "device opacity mutation mismatch at width {} lane {}",
                       width, i);
            auto expected_empty_instance =
                (i & 1u) == 0u ? i : ~0u;
            expect(static_cast<bool>(
                all(host_empty_opacity_metadata[i] ==
                    make_uint2(expected_empty_instance))))
                << luisa::format(
                       "empty-handler opacity mismatch at width {} lane {}",
                       width, i);
        }
    }
}
