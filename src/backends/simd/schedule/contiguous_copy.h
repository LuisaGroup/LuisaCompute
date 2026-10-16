#pragma once

#include <limits>
#include <string_view>

#include "schedule_ir.h"
#include <luisa/core/stl/string.h>

namespace luisa::compute::simd::schedule {

[[nodiscard]] inline luisa::string_view validate_contiguous_copy_descriptor(
    const ContiguousCopyMetadata &descriptor) noexcept {
    if (descriptor.vector_width != 2u && descriptor.vector_width != 4u && descriptor.vector_width != 8u) {
        return "contiguous copy requires a supported SIMD vector width (2, 4 or 8)";
    }
    if (descriptor.element_count == 0u ||
        descriptor.element_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 4u) {
        return "contiguous copy requires a positive representable FP32 element count";
    }
    return {};
}

[[nodiscard]] inline luisa::string_view validate_contiguous_copy(
    const ContiguousCopyMetadata &descriptor, uint64_t destination_capacity) noexcept {
    if (auto error = validate_contiguous_copy_descriptor(descriptor); !error.empty()) { return error; }
    if (descriptor.element_count > destination_capacity) {
        return "contiguous copy exceeds its destination array capacity";
    }
    return {};
}

}// namespace luisa::compute::simd::schedule
