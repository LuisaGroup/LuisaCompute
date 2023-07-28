#include <luisa/core/logging.h>
#include <luisa/dsl/polymorphic.h>

namespace luisa::compute::detail {

LC_DSL_API void polymorphic_warning_no_implementation_registered() noexcept {
    LUISA_WARNING_WITH_LOCATION("No implementations registered.");
}

LC_DSL_API void polymorphic_warning_empty_tag_group() noexcept {
    LUISA_WARNING_WITH_LOCATION("Empty tag group.");
}

LC_DSL_API void polymorphic_warning_empty_tag_range(uint lo, uint hi) noexcept {
    LUISA_WARNING_WITH_LOCATION("Empty polymorphic tag range [{}, {}).", lo, hi);
}

LC_DSL_API void polymorphic_error_unordered_tag_range(uint lo, uint hi) noexcept {
    LUISA_ERROR_WITH_LOCATION("Unordered polymorphic tag range [{}, {}).", lo, hi);
}

LC_DSL_API void polymorphic_error_overflowed_tag_range(uint lo, uint hi, uint tag_count) noexcept {
    LUISA_ERROR_WITH_LOCATION("Polymorphic tag range [{}, {}) overflowed {}.", lo, hi, tag_count);
}

}// namespace luisa::compute::detail

