#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device(argc, argv);
    auto &device = dc.device;
    auto stream = device.create_stream();
    const std::array vertices{make_float3(-2.0f, -2.0f, 0.0f),
                              make_float3(2.0f, -2.0f, 0.0f),
                              make_float3(0.0f, 2.0f, 0.0f)};
    const std::array triangles{Triangle{0u, 1u, 2u}};
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    auto scene = device.create_accel();
    scene.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);

    constexpr auto pattern_count = 16u;
    constexpr auto records_per_pattern = 6u;
    std::array<float4, pattern_count> host_inputs{};
    std::array<uint4, pattern_count * records_per_pattern> host_results;
    std::array<uint4, pattern_count> host_hits;
    host_results.fill(make_uint4(0xffffffffu));
    host_hits.fill(make_uint4(0xffffffffu));
    for (auto pattern = 0u; pattern < pattern_count; pattern++) {
        for (auto component = 0u; component < 4u; component++) {
            auto magnitude = static_cast<float>(pattern + component + 1u);
            host_inputs[pattern][component] = (pattern & (1u << component)) != 0u ? magnitude : -magnitude;
        }
    }
    auto inputs = device.create_buffer<float4>(host_inputs.size());
    auto results = device.create_buffer<uint4>(host_results.size());
    auto hits = device.create_buffer<uint4>(host_hits.size());

    Kernel1D kernel = [](AccelVar accel, BufferFloat4 inputs,
                         BufferUInt4 results, BufferUInt4 hits) noexcept {
        auto lane = dispatch_id().x;
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 2.0f);
        auto hit = accel.traverse_any(ray, {})
                       .on_surface_candidate([](SurfaceCandidate &candidate) noexcept { candidate.commit(); })
                       .trace();
        hits.write(lane, make_uint4(hit->hit_type, hit->inst, hit->prim, hit->distance().as<uint>()));
        // Runtime floating-point comparisons preserve vector predicates rather
        // than allowing the front end to fold reductions of an integer mask.
        // The bool3 cases exercise non-power-of-two predicate widths in the
        // RTX compiler; inspect the optimized IR dump to confirm that shape.
        Float4 value = inputs.read(lane);
        auto write_reductions = [&](auto predicate, uint record) noexcept {
            Bool every = all(predicate);
            Bool some = any(predicate);
            Bool every_inverted = all(!predicate);
            Bool some_inverted = any(!predicate);
            results.write(lane * 6u + record,
                          make_uint4(cast<uint>(every), cast<uint>(some),
                                     cast<uint>(every_inverted), cast<uint>(some_inverted)));
            results.write(lane * 6u + record + 1u,
                          make_uint4(cast<uint>(!every), cast<uint>(!some),
                                     cast<uint>(!every_inverted), cast<uint>(!some_inverted)));
        };
        $if (hit->hit_type == static_cast<uint>(HitType::Surface)) {
            write_reductions(value.xy() > 0.0f, 0u);
            write_reductions(value.xyz() > 0.0f, 2u);
            write_reductions(value > 0.0f, 4u);
        };
    };
    LUISA_INFO("RTX bool2/3/4 reduction fixture: AST hash={:016x}.", kernel.function()->function().hash());
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = false});
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build() << scene.build()
           << inputs.copy_from(luisa::span{host_inputs})
           << results.copy_from(luisa::span{host_results})
           << hits.copy_from(luisa::span{host_hits})
           << shader(scene, inputs, results, hits).dispatch(pattern_count)
           << results.copy_to(luisa::span{host_results})
           << hits.copy_to(luisa::span{host_hits}) << synchronize();

    auto correct = true;
    auto hits_correct = true;
    const auto expected_hit = make_uint4(static_cast<uint>(HitType::Surface), 0u, 0u, 0x3f800000u);
    for (auto pattern = 0u; pattern < pattern_count; pattern++) {
        hits_correct = hits_correct && all(host_hits[pattern] == expected_hit);
        for (auto width = 2u; width <= 4u; width++) {
            auto full_mask = (1u << width) - 1u;
            auto active_mask = pattern & full_mask;
            const auto expected = make_uint4(active_mask == full_mask, active_mask != 0u,
                                             active_mask == 0u, active_mask != full_mask);
            for (auto invert_result = 0u; invert_result < 2u; invert_result++) {
                auto expected_result = invert_result == 0u ? expected : make_uint4(1u) - expected;
                auto actual = host_results[pattern * records_per_pattern + (width - 2u) * 2u + invert_result];
                auto valid = all(actual == expected_result);
                expect(valid) << "RTX boolean reductions: width" << width << "pattern" << pattern
                              << "negated result" << invert_result;
                if (!valid) {
                    LUISA_WARNING("RTX bool{} pattern {} negated {}: got ({}, {}, {}, {}), expected ({}, {}, {}, {}).",
                                  width, pattern, invert_result, actual.x, actual.y, actual.z, actual.w,
                                  expected_result.x, expected_result.y, expected_result.z, expected_result.w);
                }
                correct = correct && valid;
            }
        }
    }
    expect(hits_correct) << "boolean reduction dispatch must retain the exact ray-query hit";
    return correct && hits_correct ? 0 : 1;
}
