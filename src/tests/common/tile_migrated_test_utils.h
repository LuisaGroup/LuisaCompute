#pragma once

#include "compute/tile/kernels.h"
#include <array>
#include <cmath>
#include <optional>

namespace luisa::test::tile_migrated {

enum class Operation {
    COPY,
    ADD,
    SAXPY,
    CLAMP,
    EXP,
    RMS_NORM,
    SUM,
    MAX,
    MIN,
    ABS_SUM,
    ABS_MAX,
    CUMSUM,
    CUMMAX,
    TRANSPOSE,
    GEMM
};

inline constexpr std::array names{
    "copy", "add", "saxpy", "clamp", "exp", "rmsnorm", "sum", "max", "min",
    "abssum", "absmax", "cumsum", "cummax", "transpose", "gemm"};

[[nodiscard]] inline std::optional<Operation> parse(string_view name) {
    for (auto i = 0u; i < names.size(); i++) {
        if (name == names[i]) { return static_cast<Operation>(i); }
    }
    return std::nullopt;
}

[[nodiscard]] inline bool is_reduction(Operation op) {
    return op >= Operation::SUM && op <= Operation::ABS_MAX;
}

struct Case {
    compute::tile::Kernel kernel;
    vector<vector<float>> inputs;
    vector<double> expected;
};

// Bounded, exactly representable inputs, also exported by the benchmark.
// Matrix oracles exploit their documented periods (97 and 89) to check EVERY
// output of a 4096^3 case without a 68-billion-iteration host scalar loop.
// This is a structured test distribution, not a random-input accuracy claim.
[[nodiscard]] inline float input_a(size_t index) {
    return static_cast<float>(static_cast<int>((index % 97u * 17u + 3u) % 97u) - 48) / 64.0f;
}
[[nodiscard]] inline float input_b(size_t index) {
    return static_cast<float>(static_cast<int>((index % 89u * 13u + 7u) % 89u) - 44) / 64.0f;
}

[[nodiscard]] inline Case make(Operation op, int64_t rows, int64_t columns, int64_t depth = 1,
                               example::tile::Block block = {}) {
    namespace kernels = example::tile;
    auto matrix = op == Operation::GEMM;
    auto pointwise = op <= Operation::EXP;
    auto capture = [&]() -> compute::tile::Kernel {
        if (pointwise) {
            return kernels::pointwise(static_cast<kernels::Pointwise>(op), rows, columns, block);
        }
        if (is_reduction(op)) {
            return kernels::row_reduce(static_cast<kernels::RowReduction>(static_cast<int>(op) - static_cast<int>(Operation::SUM)),
                                       rows, columns, 4);
        }
        switch (op) {
            case Operation::RMS_NORM: return kernels::rms_norm(rows, columns);
            case Operation::CUMSUM: return kernels::row_scan(kernels::Scan::SUM, rows, columns);
            case Operation::CUMMAX: return kernels::row_scan(kernels::Scan::MAX, rows, columns);
            case Operation::TRANSPOSE: return kernels::transpose(rows, columns, block);
            default: return kernels::gemm(rows, columns, depth, block);
        }
    };
    Case result{.kernel = capture()};
    result.inputs.resize(matrix || pointwise ? 2u : 1u);
    result.inputs[0].resize(static_cast<size_t>(rows * (matrix ? depth : columns)));
    for (auto i = size_t{0}; i < result.inputs[0].size(); i++) { result.inputs[0][i] = input_a(i); }
    if (result.inputs.size() == 2u) {
        result.inputs[1].resize(static_cast<size_t>((matrix ? depth : rows) * columns));
        for (auto i = size_t{0}; i < result.inputs[1].size(); i++) { result.inputs[1][i] = input_b(i); }
    }
    result.expected.resize(static_cast<size_t>(rows * (is_reduction(op) ? 1 : columns)));
    if (matrix) {
        std::array<double, 97u * 89u> table{};
        for (auto row = size_t{0}; row < 97u; row++) {
            for (auto column = size_t{0}; column < 89u; column++) {
                auto sum = 0.0;
                for (auto k = int64_t{0}; k < depth; k++) {
                    sum += static_cast<double>(input_a(row + k)) * input_b(static_cast<size_t>(k * columns) + column);
                }
                table[row * 89u + column] = sum;
            }
        }
        for (auto row = int64_t{0}; row < rows; row++) {
            for (auto column = int64_t{0}; column < columns; column++) {
                result.expected[row * columns + column] = table[(row * depth % 97) * 89 + column % 89];
            }
        }
        return result;
    }
    for (auto row = int64_t{0}; row < rows; row++) {
        auto sum = 0.0, squared = 0.0, absolute_sum = 0.0;
        auto maximum = -std::numeric_limits<double>::infinity();
        auto minimum = std::numeric_limits<double>::infinity();
        auto absolute_maximum = 0.0;
        for (auto column = int64_t{0}; column < columns; column++) {
            auto index = row * columns + column;
            auto a = static_cast<double>(result.inputs[0][index]);
            auto b = result.inputs.size() == 2u ? static_cast<double>(result.inputs[1][index]) : 0.0;
            sum += a;
            squared += a * a;
            absolute_sum += std::abs(a);
            maximum = std::max(maximum, a);
            minimum = std::min(minimum, a);
            absolute_maximum = std::max(absolute_maximum, std::abs(a));
            switch (op) {
                case Operation::COPY: result.expected[index] = a; break;
                case Operation::ADD: result.expected[index] = a + b; break;
                case Operation::SAXPY: result.expected[index] = a / static_cast<double>(0.4f) + b; break;
                case Operation::CLAMP: result.expected[index] = std::clamp(a, -0.5, 0.5); break;
                case Operation::EXP: result.expected[index] = std::exp(a); break;
                case Operation::CUMSUM: result.expected[index] = sum; break;
                case Operation::CUMMAX: result.expected[index] = maximum; break;
                case Operation::TRANSPOSE: result.expected[column * rows + row] = a; break;
                default: break;
            }
        }
        if (op == Operation::RMS_NORM) {
            auto scale = 1.0 / std::sqrt(squared / columns + static_cast<double>(1e-12f));
            for (auto column = int64_t{0}; column < columns; column++) {
                auto index = row * columns + column;
                result.expected[index] = result.inputs[0][index] * scale;
            }
        } else if (is_reduction(op)) {
            result.expected[row] = op == Operation::SUM ? sum : op == Operation::MAX ? maximum :
                                                            op == Operation::MIN     ? minimum :
                                                            op == Operation::ABS_SUM ? absolute_sum :
                                                                                       absolute_maximum;
        }
    }
    return result;
}

inline constexpr size_t padding = 17u;
inline constexpr float canary = -719.5f;
inline constexpr double atol = 5e-5, rtol = 5e-5;

struct Validation {
    bool passed{true};
    size_t bad_index{0u};
    double max_abs_error{0.0};
};

[[nodiscard]] inline Validation validate(span<const float> output, span<const double> expected, float guard_value = canary) {
    Validation result;
    if (output.size() != expected.size() + 2u * padding) {
        result.passed = false;
        return result;
    }
    for (auto i = size_t{0}; i < output.size(); i++) {
        auto guard = i < padding || i >= output.size() - padding;
        auto reference = guard ? static_cast<double>(guard_value) : expected[i - padding];
        auto error = std::abs(output[i] - reference);
        if (!std::isfinite(output[i]) || !std::isfinite(reference) ||
            (guard ? output[i] != guard_value : error > atol + rtol * std::abs(reference))) {
            if (result.passed) { result.bad_index = i; }
            result.passed = false;
        }
        if (!guard) { result.max_abs_error = std::max(result.max_abs_error, error); }
    }
    return result;
}

}// namespace luisa::test::tile_migrated
