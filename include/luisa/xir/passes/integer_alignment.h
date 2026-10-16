#pragma once

#include <cstddef>

#include <luisa/core/dll_export.h>

namespace luisa::compute::xir {

class Value;

/// Returns a conservative power-of-two divisor of every defined value that
/// `value` can produce, capped by `maximum_alignment`. The analysis follows
/// integer arithmetic modulo the result bit width. Unsupported operations,
/// non-integer values, and cyclic dependencies that cannot be established
/// without a fixed point conservatively return one.
///
/// A non-power-of-two cap is rounded down to its greatest power-of-two; zero
/// is treated as one. This makes the result directly usable as a guaranteed
/// byte alignment without turning malformed analysis input into an assertion.
[[nodiscard]] LUISA_XIR_API size_t
integer_value_guaranteed_alignment(
    const Value *value, size_t maximum_alignment) noexcept;

}// namespace luisa::compute::xir
