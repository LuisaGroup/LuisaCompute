#include <luisa/core/platform.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/logging.h>
#include <luisa/core/string_scratch.h>

namespace luisa {

namespace detail {

template<typename T>
void append_to_string(luisa::string &s, T x) noexcept {
    if constexpr (luisa::is_floating_point_v<T>) {
        fmt::format_to(std::back_inserter(s), FMT_STRING("{:a}"), x);
    } else {
        fmt::format_to(std::back_inserter(s), FMT_STRING("{}"), x);
    }
}

}// namespace detail

StringScratch::StringScratch(size_t reserved_size) noexcept {
    _buffer.reserve(luisa::align(reserved_size, 256u) -
                    1u /* count for the trailing zero */);
}

StringScratch::StringScratch() noexcept : StringScratch{std::min<size_t>(luisa::pagesize(), 64_k)} {}
StringScratch &StringScratch::operator<<(std::string_view s) noexcept { return _buffer.append(s), *this; }
StringScratch &StringScratch::operator<<(const char *s) noexcept { return *this << std::string_view{s}; }
StringScratch &StringScratch::operator<<(const std::string &s) noexcept { return *this << std::string_view{s}; }
StringScratch &StringScratch::operator<<(bool x) noexcept { return detail::append_to_string(_buffer, x), *this; }
StringScratch &StringScratch::operator<<(float x) noexcept { return detail::append_to_string(_buffer, x), *this; }
StringScratch &StringScratch::operator<<(double x) noexcept { return detail::append_to_string(_buffer, x), *this; }
StringScratch &StringScratch::operator<<(int x) noexcept { return detail::append_to_string(_buffer, x), *this; }
StringScratch &StringScratch::operator<<(uint x) noexcept { return detail::append_to_string(_buffer, x), *this; }
StringScratch &StringScratch::operator<<(size_t x) noexcept { return detail::append_to_string(_buffer, x), *this; }
luisa::string &StringScratch::string() & noexcept { return _buffer; }
const luisa::string &StringScratch::string() const & noexcept { return _buffer; }
luisa::string StringScratch::string() && noexcept { return std::move(_buffer); }
luisa::string_view StringScratch::string_view() const noexcept { return _buffer; }
const char *StringScratch::c_str() const noexcept { return _buffer.c_str(); }
bool StringScratch::empty() const noexcept { return _buffer.empty(); }
size_t StringScratch::size() const noexcept { return _buffer.size(); }
void StringScratch::pop_back() noexcept { _buffer.pop_back(); }
void StringScratch::clear() noexcept { _buffer.clear(); }
char StringScratch::back() const noexcept { return _buffer.back(); }

}// namespace luisa
