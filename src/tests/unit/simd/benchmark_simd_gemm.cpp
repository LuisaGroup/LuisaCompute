// Runtime benchmark for lane-affine buffer lowering in a portable GEMM.
//
// The benchmark deliberately uses the ordinary DSL kernel rather than a
// backend-private entry point. It validates one result against an independent
// CPU implementation, warms the compiled shader, and then reports steady-
// state throughput for repeated dispatches. Set
// LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER=1 for a same-binary SIMD A/B.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace luisa;
using namespace luisa::compute;

namespace {

constexpr auto matrix_size = 256u;
constexpr auto warmup_dispatch_count = 4u;
constexpr auto timed_dispatch_count = 128u;

[[nodiscard]] bool validate(
    luisa::span<const float> lhs,
    luisa::span<const float> rhs,
    luisa::span<const float> actual) noexcept {
    for (auto row = uint32_t{0u}; row < matrix_size; row++) {
        for (auto column = uint32_t{0u}; column < matrix_size; column++) {
            auto expected = 0.0;
            for (auto inner = uint32_t{0u}; inner < matrix_size; inner++) {
                expected += static_cast<double>(
                                lhs[row * matrix_size + inner]) *
                            static_cast<double>(
                                rhs[inner * matrix_size + column]);
            }
            auto observed = actual[row * matrix_size + column];
            auto tolerance = 2.0e-4 +
                             2.0e-4 * std::abs(expected);
            if (!std::isfinite(observed) ||
                std::abs(static_cast<double>(observed) - expected) >
                    tolerance) {
                LUISA_WARNING(
                    "GEMM benchmark mismatch at ({}, {}): expected {}, got {} "
                    "(tolerance {}).",
                    row, column, expected, observed, tolerance);
                return false;
            }
        }
    }
    return true;
}

}// namespace

int main(int argc, char *argv[]) {
    auto [context, device] = luisa::test::create_device(argc, argv);
    auto element_count = static_cast<size_t>(matrix_size) * matrix_size;
    luisa::vector<float> lhs(element_count);
    luisa::vector<float> rhs(element_count);
    luisa::vector<float> output(
        element_count, std::numeric_limits<float>::quiet_NaN());
    for (auto i = size_t{0u}; i < element_count; i++) {
        auto lhs_value = static_cast<int32_t>((i * 17u + 3u) % 29u) - 14;
        auto rhs_value = static_cast<int32_t>((i * 11u + 5u) % 31u) - 15;
        lhs[i] = static_cast<float>(lhs_value) * 0.03125f;
        rhs[i] = static_cast<float>(rhs_value) * 0.025f;
    }

    auto lhs_buffer = device.create_buffer<float>(element_count);
    auto rhs_buffer = device.create_buffer<float>(element_count);
    auto output_buffer = device.create_buffer<float>(element_count);
    Kernel2D gemm = [&](BufferFloat lhs_values, BufferFloat rhs_values,
                        BufferFloat output_values) noexcept {
        auto column = dispatch_id().x;
        auto row = dispatch_id().y;
        Float sum = 0.0f;
        for (auto inner : dynamic_range(matrix_size)) {
            sum += lhs_values.read(row * matrix_size + inner) *
                   rhs_values.read(inner * matrix_size + column);
        }
        output_values.write(row * matrix_size + column, sum);
    };
    auto shader = device.compile(gemm);
    auto stream = device.create_stream();
    stream << lhs_buffer.copy_from(luisa::span{lhs})
           << rhs_buffer.copy_from(luisa::span{rhs})
           << shader(lhs_buffer, rhs_buffer, output_buffer)
                  .dispatch(matrix_size, matrix_size)
           << output_buffer.copy_to(luisa::span{output})
           << synchronize();
    if (!validate(lhs, rhs, output)) { return 2; }

    for (auto i = uint32_t{0u}; i < warmup_dispatch_count; i++) {
        stream << shader(lhs_buffer, rhs_buffer, output_buffer)
                      .dispatch(matrix_size, matrix_size);
    }
    stream << synchronize();
    auto start = std::chrono::steady_clock::now();
    for (auto i = uint32_t{0u}; i < timed_dispatch_count; i++) {
        stream << shader(lhs_buffer, rhs_buffer, output_buffer)
                      .dispatch(matrix_size, matrix_size);
    }
    stream << synchronize();
    auto end = std::chrono::steady_clock::now();
    auto seconds = std::chrono::duration<double>(end - start).count();
    auto operations = 2.0 * matrix_size * matrix_size * matrix_size *
                      timed_dispatch_count;
    auto gflops = operations / seconds * 1.0e-9;
    std::cout << "simd_gemm,size=" << matrix_size
              << ",dispatches=" << timed_dispatch_count
              << ",seconds=" << seconds
              << ",gflops=" << gflops << '\n';
    return 0;
}
