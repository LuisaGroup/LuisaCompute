//
// Created by Mike Smith on 2022/12/25.
//

#include <core/logging.h>
#include <runtime/device.h>

namespace luisa::compute {

void Device::_check_no_implicit_binding(Function func, luisa::string_view shader_path) noexcept {
#ifndef NDEBUG
    for (auto &&b : func.argument_bindings()) {
        if (!holds_alternative<monostate>(b)) {
            LUISA_ERROR("Kernel {} with resource "
                        "bindings cannot be saved!",
                        shader_path);
        }
    }
#endif
}

}// namespace luisa::compute
