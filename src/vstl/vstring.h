#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <xxhash.h>
#include <iostream>
#include <string>
#include <vstl/Hash.h>
#include <vstl/Memory.h>
#include <vstl/Compare.h>
#include <vstl/string_view.h>
#include <vstl/vector.h>
namespace vstd {
using string = luisa::string;
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
inline void to_string(double Val, string &str) noexcept {
    int64 v = (int64)Val;
    IntegerToString(v, str, Val < 0);
    Val -= v;
    Val = abs(Val);
    char tempArr[12];
    str.push_back('.');
    for (auto i : range(12)) {
        Val *= 10;
        char x = (char)Val;
        Val -= x;
        tempArr[i] = (char)(x + 48);
    }
    size_t cullSize = 12;
    for (auto &&i : ptr_range(tempArr + 11, tempArr - 1, -1)) {
        if (i != '0') break;
        cullSize--;
    }
    str.append(tempArr, cullSize);
}
inline string to_string(double Val) noexcept {
    string str;
    to_string(Val, str);
    return str;
}
inline string to_string(float Val) noexcept {
    return to_string((double)Val);
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
struct hash<std::wstring> {
    inline size_t operator()(const std::wstring &str) const noexcept {
        return Hash::CharArrayHash((const char *)str.c_str(), str.size() * 2);
    }
};
template<>
struct compare<std::wstring> {
    int32 operator()(const std::wstring &a, const std::wstring &b) const noexcept {
        if (a.size() == b.size())
            return memcmp(a.data(), b.data(), a.size() * 2);
        else
            return (a.size() > b.size()) ? 1 : -1;
    }
    int32 operator()(const std::wstring &a, const std::wstring_view &b) const noexcept {
        if (a.size() == b.size())
            return memcmp(a.data(), b.data(), a.size() * 2);
        else
            return (a.size() > b.size()) ? 1 : -1;
    }
    int32 operator()(const std::wstring &a, wchar_t const *ptr) const noexcept {
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
auto &operator<<(std::wstring &s, T &&v) noexcept {
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, wchar_t>) {
        s += v;
        return s;
    } else {
        return s.append(std::forward<T>(v));
    }
}
