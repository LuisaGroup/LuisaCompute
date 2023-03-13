//
// Created by Mike on 3/12/2023.
//

#include <array>

#include <core/stl/format.h>
#include <core/logging.h>
#include <backends/common/string_scratch.h>

namespace luisa {

StringScratch::StringScratch(size_t reserved_size) noexcept {
    _buffer.reserve(reserved_size);
}

std::string_view StringScratch::view() const noexcept { return _buffer; }

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
    if (iter != s.data() + size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("No enough storage converting {} to string.", x);
    }
    return std::string_view{s.data(), size};
}

}// namespace detail

StringScratch &StringScratch::operator<<(bool x) noexcept {
    return *this << detail::to_string(x);
}

StringScratch &StringScratch::operator<<(float x) noexcept {
    auto s = detail::to_string(x);
    *this << s;
    return *this;
}

StringScratch &StringScratch::operator<<(int x) noexcept {
    return *this << detail::to_string(x);
}

StringScratch &StringScratch::operator<<(uint x) noexcept {
    return *this << detail::to_string(x);
}

StringScratch &StringScratch::operator<<(std::string_view s) noexcept {
    _buffer.append(s);
    return *this;
}

StringScratch &StringScratch::operator<<(size_t x) noexcept {
    return *this << detail::to_string(x);
}

StringScratch &StringScratch::operator<<(const char *s) noexcept {
    return *this << std::string_view{s};
}

StringScratch &StringScratch::operator<<(const std::string &s) noexcept {
    return *this << std::string_view{s};
}

void StringScratch::clear() noexcept { _buffer.clear(); }
bool StringScratch::empty() const noexcept { return _buffer.empty(); }
size_t StringScratch::size() const noexcept { return _buffer.size(); }
void StringScratch::pop_back() noexcept { _buffer.pop_back(); }
char StringScratch::back() const noexcept { return _buffer.back(); }
const char *StringScratch::c_str() const noexcept { return _buffer.c_str(); }

}// namespace luisa::compute
