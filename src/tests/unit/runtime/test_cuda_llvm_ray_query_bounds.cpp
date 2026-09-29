// Procedural commit interval regression, using the single-AABB fixture from
// test_metal_xir_air_ray_query. Every attempt is independent of traversal order.
#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

[[nodiscard]] bool test_callback_implicit_arguments(
    Device &device, Stream &stream, const ProceduralPrimitive &primitive) {
    const std::array vertices{
        make_float3(-0.5f, -0.5f, 1.0f),
        make_float3(0.5f, -0.5f, 1.0f),
        make_float3(0.0f, 0.5f, 1.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto scene = device.create_accel();
    scene.emplace_back(mesh, translation(3.0f, 0.0f, 0.0f), 0xffu, false);
    scene.emplace_back(primitive);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build() << scene.build();

    constexpr auto lanes_per_launch = 12u;
    constexpr auto launch_count = 4u;
    constexpr auto sentinel = ~0u;
    auto observed = device.create_buffer<uint4>(launch_count * lanes_per_launch);
    auto hits = device.create_buffer<uint4>(observed.size());
    luisa::vector<uint4> host_observed(observed.size());
    luisa::vector<uint4> host_hits(hits.size());
    Callable read_implicit = []() noexcept {
        return make_uint4(dispatch_size(), kernel_id());
    };
    Callable forward_implicit = [&]() noexcept { return read_implicit(); };
    Kernel3D kernel = [&](AccelVar accel, BufferUInt4 observed,
                          BufferUInt4 hits) noexcept {
        auto id = dispatch_id();
        auto size = dispatch_size();
        auto lane = id.x + size.x * (id.y + size.y * id.z);
        auto slot = kernel_id() * lanes_per_launch + lane;
        auto ray = make_ray(make_float3(ite((lane & 1u) == 0u, 3.0f, 0.0f), 0.0f, 2.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 3.0f);
        UInt4 callback_implicit = make_uint4(sentinel);
        UInt callback_count = 0u;
        auto hit = accel.traverse(ray, {})
                       .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                           callback_implicit = forward_implicit();
                           callback_count += 1u;
                           candidate.commit();
                       })
                       .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                           callback_implicit = forward_implicit();
                           callback_count += 1u;
                           candidate.commit(1.0f);
                       })
                       .trace();
        observed.write(slot, callback_implicit);
        hits.write(slot, make_uint4(hit->hit_type, hit->inst, hit->prim, callback_count));
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
    constexpr std::array batched_sizes{
        make_uint3(3u, 2u, 1u), make_uint3(0u),
        make_uint3(2u, 1u, 5u), make_uint3(1u, 3u, 4u)};
    constexpr std::array direct_sizes{
        make_uint3(2u, 2u, 2u), make_uint3(0u),
        make_uint3(0u), make_uint3(0u)};
    bool passed = true;
    for (auto batched : {true, false}) {
        // The zero-sized batch leaves a hole in kernel IDs. A subsequent
        // ordinary dispatch must reset the ID and every launch dimension.
        std::fill(host_observed.begin(), host_observed.end(), make_uint4(sentinel));
        std::fill(host_hits.begin(), host_hits.end(), make_uint4(sentinel));
        stream << observed.copy_from(luisa::span{host_observed})
               << hits.copy_from(luisa::span{host_hits});
        if (batched) {
            stream << shader(scene, observed, hits).dispatch(luisa::span{batched_sizes});
        } else {
            stream << shader(scene, observed, hits).dispatch(direct_sizes.front());
        }
        stream << observed.copy_to(luisa::span{host_observed})
               << hits.copy_to(luisa::span{host_hits}) << synchronize();
        auto &sizes = batched ? batched_sizes : direct_sizes;
        for (auto launch = 0u; launch < launch_count; launch++) {
            auto size = sizes[launch];
            auto count = size.x * size.y * size.z;
            for (auto lane = 0u; lane < lanes_per_launch; lane++) {
                auto slot = launch * lanes_per_launch + lane;
                auto expected_implicit = lane < count ? make_uint4(size, launch) : make_uint4(sentinel);
                auto surface = (lane & 1u) == 0u;
                auto expected_hit = lane < count ?
                                        make_uint4(static_cast<uint>(surface ? HitType::Surface : HitType::Procedural),
                                                   surface ? 0u : 1u, 0u, 1u) :
                                        make_uint4(sentinel);
                auto correct = all(host_observed[slot] == expected_implicit) &&
                               all(host_hits[slot] == expected_hit);
                expect(correct) << luisa::format(
                    "callback implicit arguments: batched={} launch={} lane={} observed={} expected={} hit={} expected_hit={}",
                    batched, launch, lane, host_observed[slot], expected_implicit, host_hits[slot], expected_hit);
                passed &= correct;
            }
        }
    }
    return passed;
}

[[nodiscard]] bool test_commit_bounds(Device &device) {
    constexpr auto t_min = 0.25f;
    constexpr auto t_max = 1.75f;
    constexpr auto initial_commit = 1.0f;
    constexpr auto infinity = std::numeric_limits<float>::infinity();
    const std::array attempts{
        std::nextafter(t_min, -infinity), t_min, std::nextafter(t_min, infinity),
        initial_commit, std::nextafter(t_max, -infinity), t_max,
        std::nextafter(t_max, infinity), -1.0f,
        infinity, -infinity, std::numeric_limits<float>::quiet_NaN()};
    constexpr auto finite_case_count = 8u;
    const std::array bounds{AABB{
        .packed_min = {-1.0f, -1.0f, 0.0f}, .packed_max = {1.0f, 1.0f, 2.0f}}};
    auto stream = device.create_stream();
    auto bounds_buffer = device.create_buffer<AABB>(bounds.size());
    auto primitive = device.create_procedural_primitive(bounds_buffer);
    auto scene = device.create_accel();
    scene.emplace_back(primitive);
    auto input = device.create_buffer<float>(attempts.size());
    auto summaries = device.create_buffer<uint4>(2u * attempts.size());
    auto distances = device.create_buffer<float2>(2u * attempts.size());
    luisa::vector<uint4> host_summaries(summaries.size());
    luisa::vector<float2> host_distances(distances.size());
    stream << bounds_buffer.copy_from(luisa::span{bounds})
           << input.copy_from(luisa::span{attempts})
           << primitive.build() << scene.build() << synchronize();

    bool passed = true;
    for (auto query_any : {false, true}) {
        Kernel1D kernel = [query_any](AccelVar accel, BufferFloat attempted_t,
                                      BufferUInt4 result, BufferFloat2 detail) noexcept {
            auto index = dispatch_x();
            auto ray = make_ray(make_float3(0.0f, 0.0f, 2.0f),
                                make_float3(0.0f, 0.0f, -1.0f), t_min, t_max);
            auto value = attempted_t.read(index);
            auto trace_one = [&](bool preserve, UInt output_index) noexcept {
                UInt callback_count = 0u;
                Float final_bound = -1.0f;
                auto on_candidate = [&](ProceduralCandidate &candidate) noexcept {
                    callback_count += 1u;
                    if (preserve) { candidate.commit(initial_commit); }
                    candidate.commit(value);
                    final_bound = candidate.ray()->t_max();
                };
                // Construct the query after initializing its captures:
                // traverse() appends the query statement immediately.
                auto hit = [&] {
                    if (query_any) {
                        return accel.traverse_any(ray, {}).on_procedural_candidate(on_candidate).trace();
                    }
                    return accel.traverse(ray, {}).on_procedural_candidate(on_candidate).trace();
                }();
                result.write(output_index, make_uint4(hit->hit_type, hit->inst,
                                                      hit->prim, callback_count));
                detail.write(output_index, make_float2(hit->distance(), final_bound));
            };
            trace_one(false, index * 2u);
            trace_one(true, index * 2u + 1u);
        };
        for (auto fast_math : {false, true}) {
            // Fast math permits the compiler to assume finite inputs. Exercise
            // NaN/Inf rejection in precise mode, and repeat every finite edge
            // through the default fast-math path with the same strict oracle.
            auto count = fast_math ? finite_case_count : static_cast<uint>(attempts.size());
            auto shader = device.compile(kernel, ShaderOption{
                                                     .enable_cache = false,
                                                     .enable_fast_math = fast_math});
            stream << shader(scene, input, summaries, distances).dispatch(count)
                   << summaries.copy_to(luisa::span{host_summaries})
                   << distances.copy_to(luisa::span{host_distances}) << synchronize();
            for (auto i = 0u; i < count; i++) {
                for (auto preserve : {false, true}) {
                    auto bound = preserve ? initial_commit : t_max;
                    auto accepted = std::isfinite(attempts[i]) &&
                                    attempts[i] >= t_min && attempts[i] <= bound;
                    auto has_hit = preserve || accepted;
                    auto expected_type = static_cast<uint>(has_hit ? HitType::Procedural : HitType::Miss);
                    auto expected_bound = accepted ? attempts[i] : bound;
                    auto output_index = i * 2u + static_cast<uint>(preserve);
                    auto summary = host_summaries[output_index];
                    auto detail = host_distances[output_index];
                    auto correct = summary.x == expected_type && summary.w == 1u &&
                                   detail.y == expected_bound &&
                                   (!has_hit || (summary.y == 0u && summary.z == 0u && detail.x == expected_bound));
                    expect(correct) << luisa::format(
                        "procedural bounds: any={} fast={} preserve={} case={} t={} summary={} detail={} expected_type={} expected_bound={}",
                        query_any, fast_math, preserve, i, attempts[i], summary, detail, expected_type, expected_bound);
                    passed &= correct;
                }
            }
        }
    }
    return test_callback_implicit_arguments(device, stream, primitive) && passed;
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device(argc, argv);
    return test_commit_bounds(dc.device) ? 0 : 1;
}
