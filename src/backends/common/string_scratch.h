//
// Created by Mike on 3/12/2023.
//

#pragma once

#include <core/basic_types.h>
#include <core/stl/string.h>

namespace luisa::compute {

namespace detail {

template<typename T>
[[nodiscard]] inline auto to_string(T x) noexcept {
    static thread_local std::array<char, 128u> s;
    auto [iter, size] = [x] {
        if constexpr (std::is_floating_point_v<T>) {
            return fmt::format_to_n(s.data(), s.size(), FMT_STRING("{:a}"), x);
        } else {
            return fmt::format_to_n(s.data(), s.size(), FMT_STRING("{}"), x);
        }
    }();
    assert(iter == s.data() + size && "No enough storage converting to string.");
    return std::string_view{s.data(), size};
}

}// namespace detail

class StringScratch {

private:
    luisa::string _buffer;

public:
    explicit StringScratch(size_t reserved_size) noexcept { _buffer.reserve(reserved_size); }
    StringScratch() noexcept : StringScratch{4_kb} {}
    auto &operator<<(std::string_view s) noexcept { return _buffer.append(s), *this; }
    auto &operator<<(const char *s) noexcept { return *this << std::string_view{s}; }
    auto &operator<<(const std::string &s) noexcept { return *this << std::string_view{s}; }
    auto &operator<<(bool x) noexcept { return *this << detail::to_string(x); }
    auto &operator<<(float x) noexcept { return *this << detail::to_string(x); }
    auto &operator<<(int x) noexcept { return *this << detail::to_string(x); }
    auto &operator<<(uint x) noexcept { return *this << detail::to_string(x); }
    auto &operator<<(size_t x) noexcept { return *this << detail::to_string(x); }
    [[nodiscard]] auto view() const noexcept { return luisa::string_view{_buffer}; }
    [[nodiscard]] const char *c_str() const noexcept { return _buffer.c_str(); }
    [[nodiscard]] bool empty() const noexcept { return _buffer.empty(); }
    [[nodiscard]] size_t size() const noexcept { return _buffer.size(); }
    void pop_back() noexcept { _buffer.pop_back(); }
    void clear() noexcept { _buffer.clear(); }
    [[nodiscard]] char back() const noexcept { return _buffer.back(); }
};

}// namespace luisa::compute
