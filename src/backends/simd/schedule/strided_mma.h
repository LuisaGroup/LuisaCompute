#pragma once

#include <array>
#include <limits>
#include <string_view>

#include "schedule_ir.h"

namespace luisa::compute::simd::schedule {

// This is a backend capability boundary, independent of packet lane width.
[[nodiscard]] constexpr bool supports_strided_mma_width(uint32_t width) noexcept {
    return width == 2u || width == 4u || width == 8u;
}

// Geometry/capacity validation has no dependency on Luisa's Type system.
// Typed admission and emission convert their reference capacities to elements.
[[nodiscard]] inline std::string_view validate_strided_mma(
    const StridedMmaMetadata &d, const std::array<uint64_t, 4u> &capacities) noexcept {
    if (!supports_strided_mma_width(d.vector_width)) {
        return "SIMD strided MMA requires vector width 2, 4, or 8";
    }
    auto rank = d.output_extents.size();
    if (rank == 0u || d.lhs_output_strides.size() != rank || d.rhs_output_strides.size() != rank) {
        return "strided MMA has inconsistent output rank";
    }
    for (auto capacity : capacities) {
        if (capacity == 0u) { return "strided MMA requires four nonempty references"; }
    }
    auto volume = uint64_t{1u};
    auto lhs_offset = uint64_t{0u};
    auto rhs_offset = uint64_t{0u};
    constexpr auto limit = std::numeric_limits<uint64_t>::max();
    auto accumulate = [](uint64_t &offset, uint64_t index, uint64_t stride) noexcept {
        if (stride != 0u && index > (limit - offset) / stride) { return false; }
        offset += index * stride;
        return true;
    };
    auto output_axis = size_t{0u};
    for (auto i = size_t{0u}; i < rank; i++) {
        auto extent = d.output_extents[i];
        if (extent == 0u || volume > limit / extent) { return "strided MMA output extent overflows"; }
        volume *= extent;
        if (extent > 1u) { output_axis = i; }
        if (d.contraction_extent != 0u &&
            (!accumulate(lhs_offset, extent - 1u, d.lhs_output_strides[i]) ||
             !accumulate(rhs_offset, extent - 1u, d.rhs_output_strides[i]))) {
            return "strided MMA operand offset overflows";
        }
    }
    if (volume > capacities[2u] || volume > capacities[3u]) {
        return "strided MMA seed/output reference capacity is insufficient";
    }
    if (d.contraction_extent != 0u &&
        (!accumulate(lhs_offset, d.contraction_extent - 1u, d.lhs_contraction_stride) ||
         !accumulate(rhs_offset, d.contraction_extent - 1u, d.rhs_contraction_stride) ||
         lhs_offset >= capacities[0u] || rhs_offset >= capacities[1u])) {
        return "strided MMA input reference capacity is insufficient";
    }
    switch (d.vectorization) {
        case StridedMmaVectorization::output: {
            auto lhs = d.lhs_output_strides[output_axis];
            auto rhs = d.rhs_output_strides[output_axis];
            if (d.output_extents[output_axis] < d.vector_width ||
                !((lhs == 0u && rhs == 1u) || (lhs == 1u && rhs == 0u))) {
                return "strided MMA output vectors require complementary broadcast/unit strides";
            }
            break;
        }
        case StridedMmaVectorization::contraction:
            if (!d.allow_reassociation || d.lhs_contraction_stride != 1u || d.rhs_contraction_stride != 1u) {
                return "strided MMA contraction vectors require reassociation and unit K strides";
            }
            break;
        default: return "strided MMA has an unknown vectorization mode";
    }
    return {};
}

// Schedule IR knows only opaque Type pointers. It checks geometry, arithmetic
// overflow and backend mode; concrete reference types/capacities are checked
// again at the typed XIR admission and LLVM emission boundaries.
[[nodiscard]] inline std::string_view validate_strided_mma_descriptor(const StridedMmaMetadata &d) noexcept {
    constexpr auto maximum = std::numeric_limits<uint64_t>::max();
    return validate_strided_mma(d, std::array<uint64_t, 4u>{maximum, maximum, maximum, maximum});
}

}// namespace luisa::compute::simd::schedule
