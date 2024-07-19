#pragma once
#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/device_interface.h>
namespace luisa::compute {
struct ToyCDeviceConfig : public DeviceConfigExt {
public:
    [[nodiscard]] virtual luisa::string dynamic_module_name() const = 0;
    [[nodiscard]] virtual luisa::string set_func_table_name() const {
        // An default unique function symbol;
        return "set_functable_e9f41b3d9cbc4eaea8306f531b1eb997";
    }
    virtual ~ToyCDeviceConfig() = default;
};
}// namespace luisa::compute