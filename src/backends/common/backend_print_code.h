#pragma once

#include <cstdlib>
#include <string_view>
#include <luisa/core/stl/string.h>

namespace luisa::compute {

[[nodiscard]] inline bool backend_print_code_enabled() noexcept {
    auto env = std::getenv("LUISA_DUMP_SOURCE");
    return env != nullptr && luisa::string_view{env} == "1";
}

}// namespace luisa::compute
