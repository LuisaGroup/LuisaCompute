#pragma once

#include <string>
#include <string_view>

#include <core/stl/memory.h>

namespace luisa {

using string = std::basic_string<char, std::char_traits<char>, luisa::allocator<char>>;
using u16string = std::basic_string<char16_t, std::char_traits<char16_t>, luisa::allocator<char16_t>>;
using u32string = std::basic_string<char32_t, std::char_traits<char32_t>, luisa::allocator<char32_t>>;
using u8string = std::basic_string<char8_t, std::char_traits<char8_t>, luisa::allocator<char8_t>>;
using wstring = std::basic_string<wchar_t, std::char_traits<wchar_t>, luisa::allocator<wchar_t>>;

using std::string_view;
using std::u16string_view;
using std::u32string_view;
using std::u8string_view;
using std::wstring_view;

template<typename Char, typename CharTraits>
struct basic_string_hash {
    using is_transparent = void;// to enable heterogeneous lookup
    using is_avalaunching = void;
    [[nodiscard]] uint64_t operator()(std::basic_string_view<Char, CharTraits> s) const noexcept {
        return hash64(s.data(), s.size() * sizeof(Char), hash64_default_seed);
    }
    template<typename Allocator>
    [[nodiscard]] uint64_t operator()(const std::basic_string<Char, CharTraits, Allocator> &s) const noexcept {
        return hash64(s.data(), s.size() * sizeof(Char), hash64_default_seed);
    }
    [[nodiscard]] uint64_t operator()(const Char *s) const noexcept {
        return hash64(s, CharTraits::length(s) * sizeof(Char), hash64_default_seed);
    }
};

using string_hash = basic_string_hash<char, std::char_traits<char>>;
using u16string_hash = basic_string_hash<char16_t, std::char_traits<char16_t>>;
using u32string_hash = basic_string_hash<char32_t, std::char_traits<char32_t>>;
using u8string_hash = basic_string_hash<char8_t, std::char_traits<char8_t>>;
using wstring_hash = basic_string_hash<wchar_t, std::char_traits<wchar_t>>;

template<typename Char>
struct is_char : std::disjunction<
                     std::is_same<std::remove_cvref_t<Char>, char>,
                     std::is_same<std::remove_cvref_t<Char>, char8_t>,
                     std::is_same<std::remove_cvref_t<Char>, char16_t>,
                     std::is_same<std::remove_cvref_t<Char>, char32_t>,
                     std::is_same<std::remove_cvref_t<Char>, wchar_t>> {};

template<typename Char>
constexpr bool is_char_v = is_char<Char>::value;

template<typename Char>
    requires is_char_v<Char>
struct hash<Char *> : string_hash {};

template<typename Char, size_t N>
    requires is_char_v<Char>
struct hash<Char[N]> : string_hash {};

template<typename C, typename CT, typename Alloc>
struct hash<std::basic_string<C, CT, Alloc>> : basic_string_hash<C, CT> {};

template<typename C, typename CT>
struct hash<std::basic_string_view<C, CT>> : basic_string_hash<C, CT> {};

}// namespace luisa
