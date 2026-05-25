//
// Created by Mike on 2024/5/10.
//

#include <luisa/core/logging.h>
#include <luisa/coro/coro_frame_buffer.h>

namespace luisa::compute::detail {
void error_coro_frame_buffer_invalid_element_size(size_t stride, size_t expected) noexcept {
    LUISA_ERROR(
        "Invalid coroutine frame buffer view element size {} (expected {}).",
        stride, expected);
}
}// namespace luisa::compute::detail
