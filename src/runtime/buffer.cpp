// #include <runtime/dynamic_buffer.h>
#include <core/logging.h>

namespace luisa::compute::detail {

LC_RUNTIME_API void error_buffer_copy_sizes_mismatch(size_t src, size_t dst) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Incompatible buffer views with different element counts (src = {}, dst = {}).",
        src, dst);
}

LC_RUNTIME_API void error_buffer_reinterpret_size_too_small(size_t size, size_t dst) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Unable to hold any element (with size = {}) in buffer view (with size = {}).",
        size, dst);
}

LC_RUNTIME_API void error_buffer_subview_overflow(size_t offset, size_t ele_size, size_t size) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Subview (with offset_elements = {}, size_elements = {}) "
        "overflows buffer view (with size_elements = {}).",
        offset, ele_size, size);
}

LC_RUNTIME_API void error_buffer_invalid_alignment(size_t offset, size_t dst) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Invalid buffer view offset {} for elements with alignment {}.",
        offset, dst);
}

}// namespace luisa::compute::detail
