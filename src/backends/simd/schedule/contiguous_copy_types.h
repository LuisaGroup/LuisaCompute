#pragma once

#include <array>

#include <luisa/ast/type.h>
#include <luisa/core/stl/string.h>

#include "contiguous_copy.h"

namespace luisa::compute::simd::schedule {

[[nodiscard]] inline luisa::string_view validate_contiguous_copy(
    const ContiguousCopyMetadata &descriptor,
    const std::array<const Type *, 3u> &types) noexcept {
    if (types[0u] == nullptr || !types[0u]->is_buffer() || types[0u]->element() == nullptr ||
        types[0u]->element()->tag() != Type::Tag::FLOAT32) {
        return "contiguous copy source must be a typed buffer<float> resource";
    }
    if (types[1u] == nullptr || types[1u]->tag() != Type::Tag::UINT64) {
        return "contiguous copy offset must be a uint64 value";
    }
    if (types[2u] == nullptr || !types[2u]->is_array() ||
        types[2u]->element()->tag() != Type::Tag::FLOAT32 || types[2u]->dimension() == 0u) {
        return "contiguous copy destination must be a fixed array<float> reference";
    }
    return validate_contiguous_copy(descriptor, types[2u]->dimension());
}

}// namespace luisa::compute::simd::schedule
