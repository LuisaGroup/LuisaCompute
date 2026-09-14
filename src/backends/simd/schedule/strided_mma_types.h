#pragma once

#include <luisa/ast/type.h>

#include "strided_mma.h"

namespace luisa::compute::simd::schedule {

// Deliberately separate from the dependency-light Schedule dialect. Both
// typed boundaries share this adapter and the same geometry/capacity check.
[[nodiscard]] inline std::string_view validate_strided_mma(
    const StridedMmaMetadata &d, const std::array<const Type *, 4u> &types) noexcept {
    std::array<uint64_t, 4u> capacities{};
    for (auto i = size_t{0u}; i < types.size(); i++) {
        auto *type = types[i];
        if (type == nullptr || !type->is_array() || type->element()->tag() != Type::Tag::FLOAT32 || type->dimension() == 0u) {
            return "strided MMA requires four nonempty fixed-array<float> references";
        }
        capacities[i] = type->dimension();
    }
    return validate_strided_mma(d, capacities);
}

}// namespace luisa::compute::simd::schedule
