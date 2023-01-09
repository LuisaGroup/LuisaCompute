//
// Created by Mike Smith on 2023/1/9.
//

#include <core/logging.h>
#include <runtime/volume.h>

namespace luisa::compute ::detail {

LC_RUNTIME_API void error_volume_invalid_mip_levels(size_t level, size_t mip) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Invalid mipmap level {} for volume with {} levels.",
        level, mip);
}

}// namespace luisa::compute::detail
