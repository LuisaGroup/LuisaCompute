#include <luisa/core/logging.h>
#include <luisa/dsl/local.h>

namespace luisa::compute::detail {

void local_array_error_sizes_missmatch(size_t lhs, size_t rhs) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Incompatible sizes ({} and {}).", lhs, rhs);
}

}// namespace luisa::compute::detail

