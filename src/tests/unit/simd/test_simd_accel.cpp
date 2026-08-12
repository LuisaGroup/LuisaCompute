// Test for SIMD acceleration-structure packet execution.
// This test covers:
// - W1/W2/W4/W8/W16 closest-hit and any-hit traversal
// - varying-time motion closest-hit and any-hit traversal
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
        auto accel = device.create_accel();
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

        Kernel1D kernel = [width](
                              BufferVar<Ray> ray_buffer,
                              BufferUInt mask_buffer,
                              BufferUInt4 id_buffer,
                              BufferFloat4 detail_buffer,
                              AccelVar scene,
                              AccelVar motion_scene,
                              BufferUInt4 motion_id_buffer,
                              BufferFloat4 motion_detail_buffer) noexcept {
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
        };
        auto shader = device.compile(kernel);

        std::array<uint4, thread_count> host_ids{};
        std::array<float4, thread_count> host_details{};
        std::array<uint4, thread_count> host_motion_ids{};
        std::array<float4, thread_count> host_motion_details{};
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
                      motion_accel, motion_ids, motion_details)
                      .dispatch(thread_count)
               << ids.copy_to(luisa::span{host_ids})
               << details.copy_to(luisa::span{host_details})
               << motion_ids.copy_to(luisa::span{host_motion_ids})
               << motion_details.copy_to(
                      luisa::span{host_motion_details})
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
        }
    }
}
