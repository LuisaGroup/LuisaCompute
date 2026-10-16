#pragma once

#include <limits>

#include <luisa/core/stl/memory.h>
#include <luisa/xir/passes/lower_ray_query_to_pipeline.h>

namespace luisa::compute::xir::detail {

struct RayQueryCaptureCost {
    size_t input{0u};
    size_t output{0u};

    [[nodiscard]] size_t total() const noexcept {
        auto limit = std::numeric_limits<size_t>::max();
        return input > limit - output ? limit : input + output;
    }
};

[[nodiscard]] inline size_t saturating_capture_cost_add(
    size_t total, size_t increment) noexcept {
    auto limit = std::numeric_limits<size_t>::max();
    return total > limit - increment ? limit : total + increment;
}

template<typename Input, typename Output>
[[nodiscard]] inline RayQueryCaptureCost ray_query_capture_cost(
    luisa::span<Input> inputs,
    luisa::span<Output> outputs,
    const LowerRayQueryToPipelineOptions &options) noexcept {
    RayQueryCaptureCost cost;
    if (options.captured_argument_cost == nullptr) { return cost; }
    for (auto value : inputs) {
        cost.input = saturating_capture_cost_add(
            cost.input,
            options.captured_argument_cost(value, false));
    }
    for (auto value : outputs) {
        cost.output = saturating_capture_cost_add(
            cost.output,
            options.captured_argument_cost(value, true));
    }
    return cost;
}

}// namespace luisa::compute::xir::detail
