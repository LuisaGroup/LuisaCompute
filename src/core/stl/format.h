//
// Created by Mike on 2022/9/30.
//

#pragma once

#include <spdlog/fmt/fmt.h>
#include <core/stl/string.h>

namespace luisa {

#ifndef FMT_STRING
#define FMT_STRING(...) __VA_ARGS__
#endif
template<typename String, typename Format, typename... Args>
[[nodiscard]] inline auto format(Format &&f, Args &&...args) noexcept {
    using char_type = typename String::value_type;
    using memory_buffer = fmt::basic_memory_buffer<char_type, fmt::inline_buffer_size, luisa::allocator<char_type>>;
    memory_buffer buffer;
    fmt::format_to(std::back_inserter(buffer), std::forward<Format>(f), std::forward<Args>(args)...);
    return String{buffer.data(), buffer.size()};
}
template<typename Format, typename... Args>
[[nodiscard]] inline auto format(Format &&f, Args &&...args) noexcept {
    return format<luisa::string>(std::forward<Format>(f), std::forward<Args>(args)...);
}
[[nodiscard]] inline auto hash_to_string(uint64_t hash) noexcept {
    return luisa::format("{:016X}", hash);
}

}// namespace luisa
