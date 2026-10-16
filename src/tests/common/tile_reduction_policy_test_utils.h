#pragma once

#include <array>
#include <luisa/tile.h>

namespace luisa::test::tile_reduction {

inline constexpr auto outputs = int64_t{8};

struct Subtract {
    template<typename T>
    [[nodiscard]] static constexpr T identity() noexcept { return T{0}; }
    template<typename A, typename B>
    [[nodiscard]] auto operator()(const A &a, const B &b) const noexcept { return a - b; }
};

// One source kernel is exercised by TIRx/CPU, TIRx/Metal, and XIR/SIMD.
// Coordinates belong to the logical 2-D contribution domain; the buffer's
// flattened physical storage does not define reduction order.
[[nodiscard]] inline compute::tile::Kernel folds(int64_t rows, int64_t outer, int64_t inner, float seed) {
    using namespace compute::tile;
    auto width = outer * inner;
    auto definition = tile_kernel("reduction_fold_policies", [=](TensorView<const float, 2> input,
                                                                 TensorView<float, 2> output) {
        auto a = axis("outer", outer), b = axis("inner", inner);
        auto row = axis("row_element", 1), column = axis("column", width);
        for (auto &nest : parallel(shape(rows))) {
            auto x = input[coord(nest.index(), 0), shape(row, column)];
            for (auto mode = int64_t{0}; mode < outputs - 1; mode++) {
                auto state = Scalar<float>{seed};
                auto reverse = mode == 1 || mode == 3 || mode == 4 || mode == 6;
                auto policy = reverse ? reduction::fold_right : reduction::fold_left;
                for (auto &step : nest.reduce(shape(a, b), policy)) {
                    auto ordinal = step.index(a) * inner + step.index(b);
                    auto value = x.at(coord(0, ordinal));
                    if (mode < 2) {
                        state += value;
                    } else if (mode == 3) {
                        state = value - state;
                    } else if (mode < 5) {
                        state -= value;
                    } else {
                        state = state * 2.0f + cast<float>(ordinal + 1);
                    }
                }
                output(coord(nest.index(), mode), shape(1, 1)).store(full<float>(shape(1, 1), state));
            }
            // The expression-level helper supplies the element/state order
            // appropriate for right fold; the nest itself never rewrites it.
            auto right = reduce(x, column, Subtract{}, reduction::fold_right);
            output(coord(nest.index(), outputs - 1), shape(1, 1)).store(full<float>(shape(1, 1), right.at(coord(0))));
        }
    });
    return definition.capture(tensor_shape(rows, width == 0 ? 1 : width), tensor_shape(rows, outputs));
}

[[nodiscard]] inline std::array<float, outputs> reference(span<const float> input, float seed) {
    std::array<float, outputs> result;
    result.fill(seed);
    result.back() = 0.0f;
    for (auto i = size_t{0}; i < input.size(); i++) {
        auto j = input.size() - 1u - i;
        result[0] += input[i];
        result[1] += input[j];
        result[2] -= input[i];
        result[3] = input[j] - result[3];
        result[4] -= input[j];
        result[5] = result[5] * 2.0f + static_cast<float>(i + 1u);
        result[6] = result[6] * 2.0f + static_cast<float>(j + 1u);
        result[7] = input[j] - result[7];
    }
    return result;
}

}// namespace luisa::test::tile_reduction
