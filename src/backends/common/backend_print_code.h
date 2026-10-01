#pragma once

#include <luisa/core/platform.h>

namespace luisa::compute {

[[nodiscard]] inline bool backend_print_code_enabled() noexcept {
    auto env = luisa::get_environment_variable("LUISA_DUMP_SOURCE");
    return env && luisa::string_view{*env} == "1";
}

}// namespace luisa::compute
