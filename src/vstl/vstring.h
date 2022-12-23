#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <string>
#include <vstl/hash.h>
#include <vstl/memory.h>
#include <vstl/compare.h>
#include <vstl/string_hash.h>
#include <vstl/vector.h>
#include <vstl/ranges.h>
namespace vstd {
using string = std::basic_string<char, std::char_traits<char>, luisa::allocator<char>>;
using wstring = std::basic_string<wchar_t, std::char_traits<wchar_t>, luisa::allocator<wchar_t>>;
template<class Elem, class UTy>
Elem *UIntegral_to_buff(Elem *RNext, UTy UVal) noexcept {// format UVal into buffer *ending at* RNext
    static_assert(std::is_unsigned_v<UTy>, "UTy must be unsigned");
    auto UVal_trunc = UVal;
    do {
        *--RNext = static_cast<Elem>('0' + UVal_trunc % 10);
        UVal_trunc /= 10;
    } while (UVal_trunc != 0);
    return RNext;
}

template<class Ty>
inline void IntegerToString(const Ty Val, string &str, bool negative = false) noexcept {// convert Val to string
    static_assert(std::is_integral_v<Ty>, "_Ty must be integral");
    using UTy = std::make_unsigned_t<Ty>;
    char Buff[21];// can hold -2^63 and 2^64 - 1, plus NUL
    char *const Buff_end = std::end(Buff);
    char *RNext = Buff_end;
    const auto UVal = static_cast<UTy>(Val);
    if (Val < 0 || negative) {
        RNext = UIntegral_to_buff(RNext, static_cast<UTy>(0 - UVal));
        *--RNext = '-';
    } else {
        RNext = UIntegral_to_buff(RNext, UVal);
    }
    str.append(RNext, Buff_end - RNext);
}
template<class Ty>
inline string IntegerToString(const Ty Val) noexcept {// convert Val to string
    string s;
    IntegerToString<Ty>(Val, s);
    return s;
}
inline void _float_str_resize(size_t lastSize, string &str) noexcept {
    for (int64_t i = str.size() - 1; i >= lastSize; --i) {
        if (str[i] == '.') [[unlikely]] {
            auto end = i + 2;
            int64_t j = str.size() - 1;
            for (; j >= end; --j) {
                if (str[j] != '0') {
                    break;
                }
            }
            str.resize(j + 1);
            return;
        }
    }
    str.append(".0"sv);
}
inline void to_string(double Val, string &str) noexcept {
    const size_t len = snprintf(nullptr, 0, "%f", Val);
    auto lastLen = str.size();
    str.resize(lastLen + len);
    snprintf(str.data() + lastLen, len + 1, "%f", Val);
    _float_str_resize(lastLen, str);
}

inline string to_string(double Val) noexcept {
    const size_t len = snprintf(nullptr, 0, "%f", Val);
    string str(len, '\0');
    snprintf(str.data(), len + 1, "%f", Val);
    _float_str_resize(0, str);
    return str;
}

namespace detail {

template<size_t size, bool is_signed>
struct make_integer {};

template<bool is_signed>
struct make_integer<1u, is_signed> {
    using type = std::conditional_t<is_signed, int8_t, uint8_t>;
};

template<bool is_signed>
struct make_integer<2u, is_signed> {
    using type = std::conditional_t<is_signed, int16_t, uint16_t>;
};

template<bool is_signed>
struct make_integer<4u, is_signed> {
    using type = std::conditional_t<is_signed, int32_t, uint32_t>;
};

template<bool is_signed>
struct make_integer<8u, is_signed> {
    using type = std::conditional_t<is_signed, int64_t, uint64_t>;
};

}// namespace detail

template<typename T>
using canonical_integer_t = typename detail::make_integer<sizeof(T), std::is_signed_v<T>>::type;

template<typename T>
    requires std::is_integral_v<T>
[[nodiscard]] inline auto to_string(T val) noexcept {
    return IntegerToString(static_cast<canonical_integer_t<T>>(val));
}

inline void to_string(float Val, string &str) noexcept {
    to_string((double)Val, str);
}
inline string to_string(float Val) noexcept {
    string str;
    to_string(Val, str);
    return str;
}

template<typename T>
    requires std::is_integral_v<T>
inline void to_string(T Val, string &str) noexcept {
    IntegerToString(static_cast<canonical_integer_t<T>>(Val), str);
}

}// namespace vstd

namespace vstd {
template<>
struct hash<string> {
    inline size_t operator()(string const &str) const noexcept {
        return Hash::CharArrayHash(str.data(), str.size());
    }
};

template<>
struct compare<string> {
    int32 operator()(string const &a, string const &b) const noexcept {
        if (a.size() == b.size())
            return memcmp(a.data(), b.data(), a.size());
        else
            return (a.size() > b.size()) ? 1 : -1;
    }
    int32 operator()(string const &a, const std::string_view &b) const noexcept {
        if (a.size() == b.size())
            return memcmp(a.data(), b.data(), a.size());
        else
            return (a.size() > b.size()) ? 1 : -1;
    }
    int32 operator()(string const &a, char const *ptr) const noexcept {
        size_t sz = strlen(ptr);
        if (a.size() == sz)
            return memcmp(a.data(), ptr, a.size());
        else
            return (a.size() > sz) ? 1 : -1;
    }
};
inline size_t wstrLen(wchar_t const *ptr) {
    size_t i = 0;
    while (*ptr != 0) {
        i++;
        ptr++;
    }
    return i;
}

template<>
struct hash<vstd::wstring> {
    inline size_t operator()(const vstd::wstring &str) const noexcept {
        return Hash::CharArrayHash((const char *)str.c_str(), str.size() * 2);
    }
};
template<>
struct compare<vstd::wstring> {
    int32 operator()(const vstd::wstring &a, const vstd::wstring &b) const noexcept {
        if (a.size() == b.size())
            return memcmp(a.data(), b.data(), a.size() * 2);
        else
            return (a.size() > b.size()) ? 1 : -1;
    }
    int32 operator()(const vstd::wstring &a, const std::wstring_view &b) const noexcept {
        if (a.size() == b.size())
            return memcmp(a.data(), b.data(), a.size() * 2);
        else
            return (a.size() > b.size()) ? 1 : -1;
    }
    int32 operator()(const vstd::wstring &a, wchar_t const *ptr) const noexcept {
        size_t sz = wstrLen(ptr);
        if (a.size() == sz)
            return memcmp(a.data(), ptr, a.size() * 2);
        else
            return (a.size() > sz) ? 1 : -1;
    }
};

template<>
struct compare<std::string_view> {
    int32 operator()(const std::string_view &a, string const &b) const noexcept {
        if (a.size() == b.size())
            return memcmp(a.data(), b.data(), a.size());
        else
            return (a.size() > b.size()) ? 1 : -1;
    }
    int32 operator()(const std::string_view &a, const std::string_view &b) const noexcept {
        if (a.size() == b.size())
            return memcmp(a.data(), b.data(), a.size());
        else
            return (a.size() > b.size()) ? 1 : -1;
    }
    int32 operator()(const std::string_view &a, char const *ptr) const noexcept {
        size_t sz = strlen(ptr);
        if (a.size() == sz)
            return memcmp(a.data(), ptr, a.size());
        else
            return (a.size() > sz) ? 1 : -1;
    }
};

}// namespace vstd
template<typename T>
auto &operator<<(vstd::string &s, T &&v) noexcept {
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, char>) {
        s += v;
        return s;
    } else {
        return s.append(std::forward<T>(v));
    }
}

template<typename T>
auto &operator<<(vstd::wstring &s, T &&v) noexcept {
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, wchar_t>) {
        s += v;
        return s;
    } else {
        return s.append(std::forward<T>(v));
    }
}
