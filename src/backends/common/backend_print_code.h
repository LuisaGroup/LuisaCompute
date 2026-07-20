#pragma once

#include <cstdlib>
#include <string_view>

namespace luisa::compute {

[[nodiscard]] inline bool backend_print_code_enabled() noexcept {
    static const bool enabled = [] {
        auto env = std::getenv("LUISA_DUMP_SOURCE");
        if (env == nullptr) return false;
        return std::string_view{env} == "1";
    }();
    return enabled;
}

}// namespace luisa::compute
