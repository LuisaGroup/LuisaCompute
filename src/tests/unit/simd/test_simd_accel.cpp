// Test for SIMD acceleration-structure packet execution.
// This test covers:
// - W1/W2/W4/W8/W16 closest-hit and any-hit traversal
// - varying-time motion closest-hit and any-hit traversal
// - direct varying and uniform instance metadata reads
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
        auto accel = device.create_accel({.allow_update = true});
        accel.emplace_back(
            mesh, make_float4x4(1.0f), 0x1u, true, 11u);
        accel.emplace_back(
            mesh, translation(make_float3(0.0f, 0.0f, -1.0f)),
            0x2u, true, 22u);
        auto motion_accel = device.create_accel();
        motion_accel.emplace_back(
            motion_mesh, make_float4x4(1.0f), 0x1u, true, 33u);

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
               << accel.build()
               << motion_accel.build()
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
    }
}
