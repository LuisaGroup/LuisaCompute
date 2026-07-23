#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace luisa::photon_mapping {

struct PhotonStoragePlan {
    bool valid{false};
    uint32_t path_count{0u};
    uint32_t max_depth{0u};
    uint32_t capacity{0u};
};

[[nodiscard]] constexpr PhotonStoragePlan plan_photon_storage(
    uint32_t path_count, uint32_t max_depth) noexcept {
    PhotonStoragePlan plan{
        .path_count = path_count,
        .max_depth = max_depth,
    };
    if (path_count == 0u || max_depth == 0u) { return plan; }
    auto capacity = static_cast<uint64_t>(path_count) * max_depth;
    if (capacity > std::numeric_limits<uint32_t>::max()) { return plan; }
    plan.valid = true;
    plan.capacity = static_cast<uint32_t>(capacity);
    return plan;
}

[[nodiscard]] constexpr std::optional<uint32_t> photon_slot_index(
    const PhotonStoragePlan &plan, uint32_t path_index,
    uint32_t depth) noexcept {
    if (!plan.valid || path_index >= plan.path_count ||
        depth >= plan.max_depth) {
        return std::nullopt;
    }
    return path_index * plan.max_depth + depth;
}

// The grid uses unordered atomic-exchange lists. Summing nonnegative photon
// contributions in a two-word fixed-point accumulator makes list traversal
// order irrelevant without requiring a global sort. The plan proves both the
// per-term uint32 conversion and the complete uint64 sum are representable.
struct FixedPointAccumulatorPlan {
    bool valid{false};
    uint64_t max_term_count{0u};
    uint32_t max_term_ceiling{0u};
    uint32_t fractional_bits{0u};
    uint32_t scale{0u};
    uint32_t max_quantized_term{0u};
    uint64_t max_quantized_sum{0u};
    double max_input_quantization_error{0.0};
};

[[nodiscard]] constexpr FixedPointAccumulatorPlan
plan_fixed_point_accumulator(
    uint64_t max_term_count, uint32_t max_term_ceiling,
    uint32_t fractional_bits) noexcept {
    FixedPointAccumulatorPlan plan{
        .max_term_count = max_term_count,
        .max_term_ceiling = max_term_ceiling,
        .fractional_bits = fractional_bits,
    };
    if (max_term_count == 0u || max_term_ceiling == 0u ||
        fractional_bits >= 32u) {
        return plan;
    }
    auto scale = uint64_t{1u} << fractional_bits;
    auto max_quantized_term =
        static_cast<uint64_t>(max_term_ceiling) * scale;
    if (max_quantized_term > std::numeric_limits<uint32_t>::max()) {
        return plan;
    }
    if (max_term_count >
        std::numeric_limits<uint64_t>::max() / max_quantized_term) {
        return plan;
    }
    plan.valid = true;
    plan.scale = static_cast<uint32_t>(scale);
    plan.max_quantized_term =
        static_cast<uint32_t>(max_quantized_term);
    plan.max_quantized_sum = max_term_count * max_quantized_term;
    plan.max_input_quantization_error =
        static_cast<double>(max_term_count) /
        (2.0 * static_cast<double>(scale));
    return plan;
}

struct FixedPointWords {
    uint32_t low{0u};
    uint32_t high{0u};
};

[[nodiscard]] constexpr FixedPointWords add_fixed_point_term(
    FixedPointWords sum, uint32_t term) noexcept {
    auto previous_low = sum.low;
    sum.low += term;
    sum.high += static_cast<uint32_t>(sum.low < previous_low);
    return sum;
}

[[nodiscard]] constexpr uint64_t fixed_point_word_value(
    FixedPointWords value) noexcept {
    return (static_cast<uint64_t>(value.high) << 32u) | value.low;
}

[[nodiscard]] inline std::optional<uint32_t> quantize_fixed_point_term(
    float value, const FixedPointAccumulatorPlan &plan) noexcept {
    if (!plan.valid || !std::isfinite(value) || value < 0.0f ||
        value > static_cast<float>(plan.max_term_ceiling)) {
        return std::nullopt;
    }
    auto scaled = value * static_cast<float>(plan.scale) + 0.5f;
    if (!std::isfinite(scaled) ||
        scaled > static_cast<float>(plan.max_quantized_term)) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(scaled);
}

[[nodiscard]] inline double decode_fixed_point_words(
    FixedPointWords value,
    const FixedPointAccumulatorPlan &plan) noexcept {
    if (!plan.valid) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(fixed_point_word_value(value)) /
           static_cast<double>(plan.scale);
}

}// namespace luisa::photon_mapping
