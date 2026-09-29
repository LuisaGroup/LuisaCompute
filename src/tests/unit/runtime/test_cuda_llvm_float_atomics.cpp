#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

[[nodiscard]] Float atomic_rmw(CallOp op, const BufferFloat &buffer,
                               Expr<uint> index, Expr<float> operand) noexcept {
    auto reference = buffer.atomic(index);
    switch (op) {
        case CallOp::ATOMIC_FETCH_ADD: return reference.fetch_add(operand);
        case CallOp::ATOMIC_FETCH_SUB: return reference.fetch_sub(operand);
        case CallOp::ATOMIC_FETCH_MIN: return reference.fetch_min(operand);
        case CallOp::ATOMIC_FETCH_MAX: return reference.fetch_max(operand);
        case CallOp::ATOMIC_EXCHANGE: return reference.exchange(operand);
        default: LUISA_ERROR_WITH_LOCATION("Unsupported floating-point atomic test operation.");
    }
}

[[nodiscard]] auto bits(float value) noexcept {
    return std::bit_cast<uint32_t>(value);
}

[[nodiscard]] bool run_atomics(Device &device, Stream &stream, const Accel &scene) {
    // The public DSL and AST-to-XIR atomic validator support float32. The
    // serializer's 64-bit legalization is not covered by this runtime test.
    using T = float;
    struct Case {
        CallOp operation;
        T initial;
        T operand;
        T expected;
    };
    const auto nan = std::numeric_limits<T>::quiet_NaN();
    const std::array cases{
        Case{CallOp::ATOMIC_FETCH_ADD, T{1.5}, T{2.25}, T{3.75}},
        Case{CallOp::ATOMIC_FETCH_SUB, T{5.5}, T{1.25}, T{4.25}},
        Case{CallOp::ATOMIC_FETCH_MIN, T{8.5}, T{-2.25}, T{-2.25}},
        Case{CallOp::ATOMIC_FETCH_MAX, T{-3.5}, T{6.75}, T{6.75}},
        Case{CallOp::ATOMIC_EXCHANGE, T{8.5}, T{-1.25}, T{-1.25}},
        Case{CallOp::ATOMIC_FETCH_MIN, nan, T{2.0}, T{2.0}},
        Case{CallOp::ATOMIC_FETCH_MAX, nan, T{2.0}, T{2.0}},
        Case{CallOp::ATOMIC_FETCH_MIN, T{2.0}, nan, T{2.0}},
        Case{CallOp::ATOMIC_FETCH_MAX, T{2.0}, nan, T{2.0}},
        Case{CallOp::ATOMIC_FETCH_ADD, nan, T{1.0}, nan},
        Case{CallOp::ATOMIC_EXCHANGE, T{0.0}, T{-0.0}, T{-0.0}},
        Case{CallOp::ATOMIC_EXCHANGE, T{-0.0}, T{0.0}, T{0.0}},
        Case{CallOp::ATOMIC_FETCH_ADD, T{-0.0}, T{-0.0}, T{-0.0}},
        Case{CallOp::ATOMIC_FETCH_SUB, T{-0.0}, T{0.0}, T{-0.0}},
        // Equal-sign zeros have an exact sign oracle without assuming a
        // particular minnum/maxnum choice between opposite-sign zeros.
        Case{CallOp::ATOMIC_FETCH_MIN, T{-0.0}, T{-0.0}, T{-0.0}},
        Case{CallOp::ATOMIC_FETCH_MAX, T{0.0}, T{0.0}, T{0.0}},
        Case{CallOp::ATOMIC_FETCH_MIN, nan, nan, nan},
        Case{CallOp::ATOMIC_FETCH_MAX, nan, nan, nan}};
    constexpr auto concurrent_lanes = 257u;
    auto matrix_lanes = static_cast<uint>(cases.size()) * 2u;
    Kernel1D kernel = [&cases](AccelVar accel, BufferVar<T> values, BufferVar<T> operands,
                               BufferVar<T> returned, BufferUInt4 hits, UInt concurrent) noexcept {
        set_block_size(64u);
        auto lane = dispatch_x();
        auto ray = make_ray(make_float3(0.0f, 0.0f, 1.0f),
                            make_float3(0.0f, 0.0f, -1.0f), 0.0f, 2.0f);
        auto hit = accel.traverse_any(ray, {})
                       .on_surface_candidate([](SurfaceCandidate &candidate) noexcept { candidate.commit(); })
                       .trace();
        hits.write(lane, make_uint4(hit->hit_type, hit->inst, hit->prim, hit->distance().as<uint>()));
        Var<T> selected = T{-777.0};
        $if (hit->hit_type == static_cast<uint>(HitType::Surface)) {
            $if (concurrent != 0u) {
                selected = atomic_rmw(CallOp::ATOMIC_FETCH_ADD, values, 0u, cast<T>(hit->distance()));
            }
            $else {
                $if ((lane & 1u) == 0u) {
                    for (auto i = 0u; i < cases.size(); i++) {
                        $if (lane / 2u == i) {
                            selected = atomic_rmw(cases[i].operation, values, lane, operands.read(lane));
                        };
                    }
                }
                $else {
                    selected = values.read(lane);
                };
            };
        };
        // The atomic result crosses conditional merges. Serialization-time
        // CAS expansion must preserve these PHI inputs and the old value.
        returned.write(lane, selected);
    };
    auto shader = device.compile(kernel, ShaderOption{.enable_cache = false, .enable_fast_math = false});
    auto values = device.create_buffer<T>(concurrent_lanes);
    auto operands = device.create_buffer<T>(concurrent_lanes);
    auto returned = device.create_buffer<T>(concurrent_lanes);
    auto hits = device.create_buffer<uint4>(concurrent_lanes);
    luisa::vector<T> host_values(concurrent_lanes, T{0});
    luisa::vector<T> host_operands(concurrent_lanes, T{0});
    luisa::vector<T> host_returned(concurrent_lanes);
    luisa::vector<uint4> host_hits(concurrent_lanes);
    for (auto lane = 0u; lane < matrix_lanes; lane++) {
        host_values[lane] = cases[lane / 2u].initial;
        host_operands[lane] = cases[lane / 2u].operand;
    }
    stream << values.copy_from(luisa::span{host_values})
           << operands.copy_from(luisa::span{host_operands})
           << returned.copy_from(luisa::span{host_returned})
           << hits.copy_from(luisa::span{host_hits})
           << shader(scene, values, operands, returned, hits, 0u).dispatch(matrix_lanes)
           << values.copy_to(luisa::span{host_values})
           << returned.copy_to(luisa::span{host_returned})
           << hits.copy_to(luisa::span{host_hits}) << synchronize();
    const auto expected_hit = make_uint4(static_cast<uint>(HitType::Surface), 0u, 0u, 0x3f800000u);
    auto matrix_correct = true;
    for (auto lane = 0u; lane < matrix_lanes; lane++) {
        auto initial = cases[lane / 2u].initial;
        auto expected = (lane & 1u) == 0u ? cases[lane / 2u].expected : initial;
        auto value_correct = std::isnan(expected) ? std::isnan(host_values[lane]) : bits(host_values[lane]) == bits(expected);
        if (!all(host_hits[lane] == expected_hit) || bits(host_returned[lane]) != bits(initial) || !value_correct) {
            LUISA_WARNING("RTX atomic {}-bit matrix lane {}: old={:x}/{:x}, final={:x}/{:x}, hit={}.",
                          sizeof(T) * 8u, lane, bits(host_returned[lane]), bits(initial),
                          bits(host_values[lane]), bits(expected), host_hits[lane]);
            matrix_correct = false;
        }
    }
    expect(matrix_correct) << "RTX atomic old values, conditional PHIs, quiet NaNs and signed zeros" << sizeof(T) * 8u;

    std::fill(host_values.begin(), host_values.end(), T{0});
    stream << values.copy_from(luisa::span{host_values})
           << shader(scene, values, operands, returned, hits, 1u).dispatch(concurrent_lanes)
           << values.copy_to(luisa::span{host_values})
           << returned.copy_to(luisa::span{host_returned})
           << hits.copy_to(luisa::span{host_hits}) << synchronize();
    auto concurrent_correct = std::all_of(host_returned.begin(), host_returned.end(), [](T value) noexcept { return std::isfinite(value); });
    if (concurrent_correct) { std::sort(host_returned.begin(), host_returned.end()); }
    concurrent_correct = concurrent_correct && host_values[0] == static_cast<T>(concurrent_lanes);
    for (auto lane = 0u; lane < concurrent_lanes; lane++) {
        concurrent_correct = concurrent_correct && host_returned[lane] == static_cast<T>(lane) &&
                             all(host_hits[lane] == expected_hit);
    }
    expect(concurrent_correct) << "contended RTX fetch_add returns every old value 0..256 exactly once" << sizeof(T) * 8u;
    return matrix_correct && concurrent_correct;
}

}// namespace

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
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build() << scene.build() << synchronize();
    return run_atomics(device, stream, scene) ? 0 : 1;
}
