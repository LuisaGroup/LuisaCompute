#pragma once

#include <cstdlib>
#include <string_view>

namespace luisa::compute {

[[nodiscard]] inline bool backend_print_code_enabled() noexcept {
    auto env = std::getenv("LUISA_DUMP_SOURCE");
    return env != nullptr && std::string_view{env} == "1";
}

}// namespace luisa::compute
