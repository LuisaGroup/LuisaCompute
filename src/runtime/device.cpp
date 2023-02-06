//
// Created by Mike Smith on 2022/12/25.
//

#include <core/logging.h>
#include <runtime/device.h>

namespace luisa::compute {

#ifndef NDEBUG
void Device::_check_no_implicit_binding(Function func, luisa::string_view shader_path) noexcept {
    for (auto &&b : func.argument_bindings()) {
        if (!holds_alternative<monostate>(b)) {
            LUISA_ERROR("Kernel {} with resource bindings cannot be saved!", shader_path);
        }
    }
}
#endif

#ifdef LC_ENABLE_API
uint64_t DeviceInterface::create_shader_ex(const LCKernelModule *kernel, std::string_view meta_options) noexcept {
    LUISA_ERROR_WITH_LOCATION("Should not be called.");
}
#endif

}// namespace luisa::compute
