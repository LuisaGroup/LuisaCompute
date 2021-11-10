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
inline void IntegerToString(const Ty Val, std::string &str, bool negative = false) noexcept {// convert Val to std::string
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
inline std::string IntegerToString(const Ty Val) noexcept {// convert Val to std::string
    std::string s;
    IntegerToString<Ty>(Val, s);
    return s;
}
inline void to_string(double Val, std::string &str) noexcept {
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
inline std::string to_string(double Val) noexcept {
    std::string str;
    to_string(Val, str);
    return str;
}
inline std::string to_string(float Val) noexcept {
    return to_string((double)Val);
}

inline std::string to_string(int32_t Val) noexcept {
    return IntegerToString(Val);
}
inline std::string to_string(uint32_t Val) noexcept {
    return IntegerToString(Val);
}
inline std::string to_string(int16_t Val) noexcept {
    return IntegerToString(Val);
}
inline std::string to_string(uint16_t Val) noexcept {
    return IntegerToString(Val);
}
inline std::string to_string(int8_t Val) noexcept {
    return IntegerToString(Val);
}
inline std::string to_string(uint8_t Val) noexcept {
    return IntegerToString(Val);
}
inline std::string to_string(int64_t Val) noexcept {
    return IntegerToString(Val);
}
inline std::string to_string(uint64_t Val) noexcept {
    return IntegerToString(Val);
}

inline void to_string(float Val, std::string &str) noexcept {
    to_string((double)Val, str);
}

inline void to_string(int32_t Val, std::string &str) noexcept {
    IntegerToString(Val, str);
}
inline void to_string(uint32_t Val, std::string &str) noexcept {
    IntegerToString(Val, str);
}
inline void to_string(int16_t Val, std::string &str) noexcept {
    IntegerToString(Val, str);
}
inline void to_string(uint16_t Val, std::string &str) noexcept {
    IntegerToString(Val, str);
}
inline void to_string(int8_t Val, std::string &str) noexcept {
    IntegerToString(Val, str);
}
inline void to_string(uint8_t Val, std::string &str) noexcept {
    IntegerToString(Val, str);
}
inline void to_string(int64_t Val, std::string &str) noexcept {
    IntegerToString(Val, str);
}
inline void to_string(uint64_t Val, std::string &str) noexcept {
    IntegerToString(Val, str);
}

}// namespace vstd

namespace vstd {
template<>
struct hash<std::string> {
    inline size_t operator()(std::string const &str) const noexcept {
        return Hash::CharArrayHash(str.data(), str.size());
    }
};

template<>
struct compare<std::string> {
    int32 operator()(std::string const &a, std::string const &b) const noexcept {
        if (a.size() == b.size())
            return memcmp(a.data(), b.data(), a.size());
        else
            return (a.size() > b.size()) ? 1 : -1;
    }
    int32 operator()(std::string const &a, const std::string_view &b) const noexcept {
        if (a.size() == b.size())
            return memcmp(a.data(), b.data(), a.size());
        else
            return (a.size() > b.size()) ? 1 : -1;
    }
    int32 operator()(std::string const &a, char const *ptr) const noexcept {
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
    int32 operator()(const std::string_view &a, std::string const &b) const noexcept {
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
auto &operator<<(std::string &s, T &&v) noexcept {
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
