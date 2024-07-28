#pragma once
#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/device_interface.h>
namespace luisa::compute {
struct ToyCDeviceConfig : public DeviceConfigExt {
public:
    struct FuncTable {
        void *(*persist_malloc)(size_t);
        void *(*temp_malloc)(size_t);
        void (*persist_free)(void *);
        void (*push_print_str)(char const *ptr, uint64_t len);
        void (*push_print_value)(void *value, uint32_t type);
        void (*print)();
    };
    [[nodiscard]] virtual luisa::string dynamic_module_name() const = 0;
    [[nodiscard]] virtual luisa::string set_func_table_name() const {
        // An default unique function symbol;
        return "set_functable_e9f41b3d9cbc4eaea8306f531b1eb997";
    }
    [[nodiscard]] virtual luisa::optional<FuncTable> get_functable() {
        return {};
    }
    virtual ~ToyCDeviceConfig() = default;
};
}// namespace luisa::compute