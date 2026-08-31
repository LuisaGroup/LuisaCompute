//
// Created by Mike on 2024/12/10.
//

#include <luisa/core/intrin.h>
#include "fallback_device_api_ir_module.h"
#include "fallback_device_api_wrappers_embedded.h"

namespace luisa::compute::fallback {
luisa::string_view fallback_backend_device_builtin_module() noexcept {
    return {
        reinterpret_cast<const char *>(
            luisa_compute_fallback_device_api_wrappers),
        luisa_compute_fallback_device_api_wrappers_size};
}
}// namespace luisa::compute::fallback
