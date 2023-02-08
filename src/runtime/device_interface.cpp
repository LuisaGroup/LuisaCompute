//
// Created by Mike Smith on 2023/2/8.
//

#include <core/logging.h>
#include <runtime/device_interface.h>

namespace luisa::compute {

uint64_t DeviceInterface::create_shader_ex(const ir::KernelModule *kernel, std::string_view meta_options) noexcept {
    LUISA_ERROR_WITH_LOCATION("Should not be called.");
}

}// namespace luisa::compute
