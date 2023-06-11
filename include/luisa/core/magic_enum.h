#pragma once

#include <magic_enum.hpp>
#include <type_traits>
namespace luisa {

template<typename T>
    requires std::is_enum_v<T>
[[nodiscard]] inline auto to_string(T e) noexcept {
    return magic_enum::enum_name(e);
}

}// namespace luisa

