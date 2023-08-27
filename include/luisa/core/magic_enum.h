#pragma once

#include <type_traits>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/optional.h>

#define MAGIC_ENUM_USING_ALIAS_OPTIONAL using luisa::optional;
#define MAGIC_ENUM_USING_ALIAS_STRING using luisa::string;
#define MAGIC_ENUM_USING_ALIAS_STRING_VIEW using luisa::string_view;
#include <magic_enum.hpp>

namespace luisa {

template<typename T>
    requires std::is_enum_v<T>
[[nodiscard]] inline auto to_string(T e) noexcept {
    return magic_enum::enum_name(e);
}

}// namespace luisa
