#pragma once

// Shared finite-FP32 ranking fixture. The public topk/sort library composition
// is quadratic, not a tuned sorting/selection implementation or planner policy.
#include <luisa/core/logging.h>
#include <luisa/tile/algorithms.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>

namespace luisa::test::tile_rank {

enum class Operation { TOPK,
                       SORT };
enum class InputPattern { DUPLICATES,
                          UNIQUE,
                          ALL_EQUAL,
                          SIGNED_ZERO };

struct Case {
    compute::tile::Kernel kernel;
    int64_t rows;
    int64_t columns;
    int64_t count;
    bool descending;
    vector<float> input;
    vector<float> expected_values;
    vector<int64_t> expected_indices;
};

inline void populate(Case &fixture, InputPattern pattern = InputPattern::DUPLICATES) {
    fixture.input.resize(static_cast<size_t>(fixture.rows * fixture.columns));
    fixture.expected_values.resize(static_cast<size_t>(fixture.rows * fixture.count));
    fixture.expected_indices.resize(fixture.expected_values.size());
    vector<int64_t> order(static_cast<size_t>(fixture.columns));
    for (auto row = int64_t{0}; row < fixture.rows; row++) {
        for (auto column = int64_t{0}; column < fixture.columns; column++) {
            auto value = static_cast<float>((column * 37 + row * 17) % 31 - 15) * .25f;
            if (pattern == InputPattern::UNIQUE) {
                value = static_cast<float>(fixture.columns - column - 1) * .25f;
            } else if (pattern == InputPattern::ALL_EQUAL) {
                value = -2.5f;
            } else if (pattern == InputPattern::SIGNED_ZERO) {
                value = (column + row) % 2 == 0 ? 0.0f : -0.0f;
            }
            fixture.input[static_cast<size_t>(row * fixture.columns + column)] = value;
        }
        std::iota(order.begin(), order.end(), int64_t{0});
        // A total comparator makes the oracle independent of the host sort's
        // stability. Signed zero compares equal and keeps original index order.
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            auto x = fixture.input[static_cast<size_t>(row * fixture.columns + a)];
            auto y = fixture.input[static_cast<size_t>(row * fixture.columns + b)];
            return x == y ? a < b : fixture.descending ? x > y :
                                                         x < y;
        });
        for (auto rank = int64_t{0}; rank < fixture.count; rank++) {
            auto index = order[static_cast<size_t>(rank)];
            auto output = static_cast<size_t>(row * fixture.count + rank);
            fixture.expected_indices[output] = index;
            fixture.expected_values[output] = fixture.input[static_cast<size_t>(row * fixture.columns + index)];
        }
    }
}

[[nodiscard]] inline Case rows(Operation operation, int64_t row_count, int64_t columns,
                               int64_t count, bool descending = true) {
    using namespace compute::tile;
    constexpr auto max_elements = int64_t{1} << 26;
    LUISA_ASSERT(row_count > 0 && columns > 0 && count > 0 && count <= columns &&
                     row_count <= 65536 && columns <= 65536 && row_count <= max_elements / columns &&
                     (operation != Operation::SORT || count == columns),
                 "Invalid finite ranking shape or sort K (must equal N)");
    auto definition = tile_kernel("ranking_rows", [=](TensorView<const float, 2> X,
                                                      TensorView<float, 2> Values,
                                                      TensorView<int64_t, 2> Indices) {
        auto local_row = axis("local_row", 1);
        auto column = axis("column", columns);
        for (auto &nest : parallel(shape(row_count))) {
            auto origin = coord(nest.index(), 0);
            auto x = X.tile(origin, shape(local_row, column)).load();
            auto ranked = operation == Operation::SORT ? compute::tile::sort(x, column, descending) :
                                                         topk(x, column, static_cast<uint64_t>(count), descending);
            Values(origin, ranked.values.space()).store(ranked.values);
            Indices(origin, ranked.indices.space()).store(ranked.indices);
        }
    });
    Case result{definition.capture(tensor_shape(row_count, columns), tensor_shape(row_count, count),
                                   tensor_shape(row_count, count)),
                row_count, columns, count, descending};
    populate(result);
    return result;
}

// Three separately padded allocations exercise nonzero Runtime view offsets.
// Input data and input guards are checked after execution, not just outputs.
struct GuardedData {
    static constexpr size_t pad = 17u;
    static constexpr float input_guard = 917.25f;
    static constexpr float value_guard = -719.5f;
    static constexpr int64_t index_guard = std::numeric_limits<int64_t>::min() + 37;
    vector<float> input;
    vector<float> values;
    vector<int64_t> indices;

    explicit GuardedData(const Case &fixture)
        : input(fixture.input.size() + 2u * pad, input_guard),
          values(fixture.expected_values.size() + 2u * pad, value_guard),
          indices(fixture.expected_indices.size() + 2u * pad, index_guard) {
        std::copy(fixture.input.begin(), fixture.input.end(), input.begin() + pad);
        std::fill(values.begin() + pad, values.end() - pad, std::numeric_limits<float>::quiet_NaN());
        std::fill(indices.begin() + pad, indices.end() - pad, int64_t{-1});
    }
};

struct Validation {
    size_t value_mismatches{0u};
    size_t index_mismatches{0u};
    size_t input_mismatches{0u};
    size_t guard_mismatches{0u};
    [[nodiscard]] bool passed() const noexcept {
        return value_mismatches == 0u && index_mismatches == 0u && input_mismatches == 0u && guard_mismatches == 0u;
    }
};

[[nodiscard]] inline Validation validate(const Case &fixture, const GuardedData &data) {
    LUISA_ASSERT(data.input.size() == fixture.input.size() + 2u * GuardedData::pad &&
                     data.values.size() == fixture.expected_values.size() + 2u * GuardedData::pad &&
                     data.indices.size() == fixture.expected_indices.size() + 2u * GuardedData::pad,
                 "ranking validation requires complete padded allocations");
    Validation result;
    auto same_bits = [](float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); };
    for (auto i = size_t{0}; i < fixture.input.size(); i++) {
        result.input_mismatches += !same_bits(data.input[i + GuardedData::pad], fixture.input[i]);
    }
    for (auto i = size_t{0}; i < fixture.expected_values.size(); i++) {
        // Ranking copies a source value without FP arithmetic. Exact bits
        // additionally verify that the selected signed zero is preserved.
        result.value_mismatches += !same_bits(data.values[i + GuardedData::pad], fixture.expected_values[i]);
        result.index_mismatches += data.indices[i + GuardedData::pad] != fixture.expected_indices[i];
    }
    for (auto i = size_t{0}; i < GuardedData::pad; i++) {
        result.guard_mismatches += !same_bits(data.input[i], GuardedData::input_guard);
        result.guard_mismatches += !same_bits(data.input[data.input.size() - 1u - i], GuardedData::input_guard);
        result.guard_mismatches += !same_bits(data.values[i], GuardedData::value_guard);
        result.guard_mismatches += !same_bits(data.values[data.values.size() - 1u - i], GuardedData::value_guard);
        result.guard_mismatches += data.indices[i] != GuardedData::index_guard;
        result.guard_mismatches += data.indices[data.indices.size() - 1u - i] != GuardedData::index_guard;
    }
    return result;
}

}// namespace luisa::test::tile_rank
