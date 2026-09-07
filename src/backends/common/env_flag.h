#pragma once

#include <cstdlib>
#include <string_view>

namespace luisa::compute::detail {

// Single convention for boolean environment flags across the backends: the
// flag is enabled when its value is a truthy string ("1", "true", "TRUE",
// "on", "ON"); unset or any other value disables it.
[[nodiscard]] inline bool env_flag(const char *name) noexcept {
    auto *value = std::getenv(name);
    if (value == nullptr) { return false; }
    auto flag = std::string_view{value};
    return flag == "1" || flag == "true" || flag == "TRUE" ||
           flag == "on" || flag == "ON";
}

}// namespace luisa::compute::detail
