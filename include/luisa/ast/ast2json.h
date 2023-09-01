//
// Created by Mike on 8/29/2023.
//

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/ast/function.h>

namespace luisa::compute {
[[nodiscard]] LC_AST_API luisa::string to_json(Function function) noexcept;
}// namespace luisa::compute
