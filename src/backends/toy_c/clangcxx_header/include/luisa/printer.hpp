#pragma once
#include "type_traits/concepts.hpp"
#include "resources.hpp"
namespace luisa::shader {
template<concepts::string_literal Str, concepts::arithmetic... Args>
[[callop("device_log")]] void device_log(Str&& fmt, Args... args);
template<concepts::arithmetic... Args>
[[ext_call("device_log_ext")]] void device_log(BufferView<int8> string_buffer, Args... args);
template<concepts::arithmetic... Args>
[[ext_call("device_log_ext")]] void device_log(StringView string_buffer, Args... args);
}// namespace luisa::shader