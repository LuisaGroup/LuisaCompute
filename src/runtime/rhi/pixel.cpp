//
// Created by Mike Smith on 2023/1/9.
//

#include <core/logging.h>
#include <runtime/rhi/pixel.h>

namespace luisa::compute::detail {

void error_pixel_invalid_format(const char *name) noexcept {
    LUISA_ERROR_WITH_LOCATION("Invalid pixel storage for {} format.", name);
}

}// namespace luisa::compute::detail
