#pragma once

#include <luisa/core/platform.h>

namespace luisa::compute::detail {

// Single convention for boolean environment flags across the backends: the
// flag is enabled when its value is a truthy string ("1", "true", "TRUE",
// "on", "ON"); unset or any other value disables it.
[[nodiscard]] inline bool env_flag(const char *name) noexcept {
    auto value = luisa::get_environment_variable(name);
    if (!value) { return false; }
    auto flag = luisa::string_view{*value};
    return flag == "1" || flag == "true" || flag == "TRUE" ||
           flag == "on" || flag == "ON";
}

}// namespace luisa::compute::detail
