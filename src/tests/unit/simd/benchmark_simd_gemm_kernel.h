#pragma once

#include <luisa/dsl/sugar.h>

namespace luisa::compute::simd::benchmark {

constexpr auto gemm_matrix_size = uint32_t{256u};

[[nodiscard]] inline auto make_gemm_kernel() noexcept {
    return Kernel2D{[](BufferFloat lhs_values, BufferFloat rhs_values,
                       BufferFloat output_values) noexcept {
        auto column = dispatch_id().x;
        auto row = dispatch_id().y;
        Float sum = 0.0f;
        for (auto inner : dynamic_range(gemm_matrix_size)) {
            sum += lhs_values.read(row * gemm_matrix_size + inner) *
                   rhs_values.read(inner * gemm_matrix_size + column);
        }
        output_values.write(row * gemm_matrix_size + column, sum);
    }};
}

}// namespace luisa::compute::simd::benchmark
