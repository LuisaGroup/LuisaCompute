#pragma once
#include "type_traits/concepts.hpp"
namespace luisa::shader {
#ifdef DEBUG
template<concepts::string_literal Str, typename... Args>
[[callop("device_log")]] void device_log(Str &&fmt, Args... args);
#else
#define device_log(...) (0)
#endif
}// namespace luisa::shader