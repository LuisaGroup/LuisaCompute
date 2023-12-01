#include <luisa/core/logging.h>
#include <luisa/runtime/volume.h>

namespace luisa::compute ::detail {

LC_RUNTIME_API void error_volume_invalid_mip_levels(size_t level, size_t mip) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Invalid mipmap level {} for volume with {} levels.",
        level, mip);
}

LC_RUNTIME_API void volume_size_zero_error() noexcept {
    LUISA_ERROR_WITH_LOCATION("Volume size must be non-zero.");
}

}// namespace luisa::compute::detail

