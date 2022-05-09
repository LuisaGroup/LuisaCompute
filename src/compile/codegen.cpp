//
// Created by Mike Smith on 2021/3/5.
//

#include <core/logging.h>
#include <ast/type_registry.h>
#include <compile/codegen.h>

namespace luisa::compute {

Codegen::Scratch::Scratch() noexcept { _buffer.reserve(4096u - 1u); }
std::string_view Codegen::Scratch::view() const noexcept { return _buffer; }

namespace detail {

template<typename T>
[[nodiscard]] inline auto to_string(T x) noexcept {
    static thread_local std::array<char, 128u> s;
    auto [iter, size] = fmt::format_to_n(s.data(), s.size(), FMT_STRING("{}"), x);
    if (iter != s.data() + size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("No enough storage converting {} to string.", x);
    }
    return std::string_view{s.data(), size};
}

}// namespace detail

Codegen::Scratch &Codegen::Scratch::operator<<(bool x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(float x) noexcept {
    auto s = detail::to_string(x);
    *this << s;
    if (s.find('.') == std::string_view::npos && s.find('e') == std::string_view::npos) { *this << ".0"; }
    return *this;
}

Codegen::Scratch &Codegen::Scratch::operator<<(int x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(uint x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(std::string_view s) noexcept {
    _buffer.append(s);
    return *this;
}

Codegen::Scratch &Codegen::Scratch::operator<<(size_t x) noexcept {
    return *this << detail::to_string(x);
}

Codegen::Scratch &Codegen::Scratch::operator<<(const char *s) noexcept {
    return *this << std::string_view{s};
}

Codegen::Scratch &Codegen::Scratch::operator<<(const std::string &s) noexcept {
    return *this << std::string_view{s};
}

void Codegen::Scratch::clear() noexcept { _buffer.clear(); }
bool Codegen::Scratch::empty() const noexcept { return _buffer.empty(); }
size_t Codegen::Scratch::size() const noexcept { return _buffer.size(); }
void Codegen::Scratch::pop_back() noexcept { _buffer.pop_back(); }
char Codegen::Scratch::back() const noexcept { return _buffer.back(); }
const char *Codegen::Scratch::c_str() const noexcept { return _buffer.c_str(); }

}// namespace luisa::compute
