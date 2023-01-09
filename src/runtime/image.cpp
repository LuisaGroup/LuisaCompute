#include <core/logging.h>
#include <runtime/image.h>

namespace luisa::compute ::detail {

LC_RUNTIME_API void error_image_invalid_mip_levels(size_t level, size_t mip) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Invalid mipmap level {} for image with {} levels.",
        level, mip);
}

}// namespace luisa::compute::detail
