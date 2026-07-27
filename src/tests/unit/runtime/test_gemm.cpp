// Test a small non-square general matrix multiplication kernel.
//
// The deliberately different M, N, and K dimensions catch row/column stride
// mistakes. Every output is compared with an independent double-precision CPU
// implementation.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <cmath>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

void test_gemm(Device &device) {
    constexpr auto m = 7u;
    constexpr auto n = 9u;
    constexpr auto k = 5u;

    luisa::vector<float> lhs(m * k);
    luisa::vector<float> rhs(k * n);
    for (auto row = 0u; row < m; row++) {
        for (auto column = 0u; column < k; column++) {
            auto value = static_cast<int>((row * 11u + column * 7u) % 19u) - 9;
            lhs[row * k + column] = static_cast<float>(value) * 0.125f;
        }
    }
    for (auto row = 0u; row < k; row++) {
        for (auto column = 0u; column < n; column++) {
            auto value = static_cast<int>((row * 5u + column * 13u) % 23u) - 11;
            rhs[row * n + column] = static_cast<float>(value) * 0.0625f;
        }
    }

    luisa::vector<float> expected(m * n);
    for (auto row = 0u; row < m; row++) {
        for (auto column = 0u; column < n; column++) {
            double sum = 0.0;
            for (auto inner = 0u; inner < k; inner++) {
                sum += static_cast<double>(lhs[row * k + inner]) *
                       static_cast<double>(rhs[inner * n + column]);
            }
            expected[row * n + column] = static_cast<float>(sum);
        }
    }

    auto lhs_buffer = device.create_buffer<float>(lhs.size());
    auto rhs_buffer = device.create_buffer<float>(rhs.size());
    auto result_buffer = device.create_buffer<float>(expected.size());
    Kernel2D gemm = [&](BufferFloat lhs_values, BufferFloat rhs_values,
                        BufferFloat result_values) noexcept {
        auto column = dispatch_id().x;
        auto row = dispatch_id().y;
        Float sum = 0.0f;
        for (auto inner : dynamic_range(k)) {
            sum += lhs_values.read(row * k + inner) *
                   rhs_values.read(inner * n + column);
        }
        result_values.write(row * n + column, sum);
    };
    auto shader = device.compile(gemm);

    luisa::vector<float> actual(expected.size(),
                                std::numeric_limits<float>::quiet_NaN());
    auto stream = device.create_stream();
    stream << lhs_buffer.copy_from(luisa::span{lhs})
           << rhs_buffer.copy_from(luisa::span{rhs})
           << shader(lhs_buffer, rhs_buffer, result_buffer).dispatch(n, m)
           << result_buffer.copy_to(luisa::span{actual})
           << synchronize();

    size_t mismatch_count = 0u;
    for (size_t i = 0u; i < expected.size(); i++) {
        auto tolerance = 1.0e-6f + 1.0e-6f * std::abs(expected[i]);
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) > tolerance) {
            if (mismatch_count < 8u) {
                auto row = i / n;
                auto column = i % n;
                LUISA_WARNING("GEMM mismatch at ({}, {}): expected {}, got {} "
                              "(tolerance {}).",
                              row, column, expected[i], actual[i], tolerance);
            }
            mismatch_count++;
        }
    }
    expect(mismatch_count == 0u)
        << luisa::format("GEMM produced {} mismatches out of {} outputs",
                         mismatch_count, expected.size());
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    test_gemm(dc->device);
}
