#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <xhash>
#include <iostream>
#include <string>
#include <vstl/Hash.h>
#include <vstl/Memory.h>
#include <vstl/Compare.h>
#include <vstl/string_view.h>
#include <vstl/vector.h>
namespace vstd {
class VENGINE_DLL_COMMON string_core {
public:
    char *_ptr;
    size_t _lenSize;
    static constexpr size_t PLACEHOLDERSIZE = 16;
    union {
        char localStorage[PLACEHOLDERSIZE];
        size_t _capacity;
    };
    string_core() {
        _lenSize = 0;
        _ptr = localStorage;
    }
    bool IsSmall() const;
    void operator=(string_core const &v);
    string_core(string_core &&v);
    size_t capacity() const;
    ~string_core();
    void reserve(size_t tarCapa);
};
template<class _Elem, class _UTy>
_Elem *UIntegral_to_buff(_Elem *_RNext, _UTy _UVal) noexcept {// format _UVal into buffer *ending at* _RNext
    static_assert(std::is_unsigned_v<_UTy>, "_UTy must be unsigned");

#ifdef _WIN64
    auto _UVal_trunc = _UVal;
#else // ^^^ _WIN64 ^^^ // vvv !_WIN64 vvv

    constexpr bool _Big_uty = sizeof(_UTy) > 4;
    if _CONSTEXPR_IF (_Big_uty) {// For 64-bit numbers, work in chunks to avoid 64-bit divisions.
        while (_UVal > 0xFFFFFFFFU) {
            auto _UVal_chunk = static_cast<unsigned long>(_UVal % 1000000000);
            _UVal /= 1000000000;

            for (int32_t _Idx = 0; _Idx != 9; ++_Idx) {
                *--_RNext = static_cast<_Elem>('0' + _UVal_chunk % 10);
                _UVal_chunk /= 10;
            }
        }
    }

    auto _UVal_trunc = static_cast<unsigned long>(_UVal);
#endif// _WIN64

    do {
        *--_RNext = static_cast<_Elem>('0' + _UVal_trunc % 10);
        _UVal_trunc /= 10;
    } while (_UVal_trunc != 0);
    return _RNext;
}
template<class _Ty>
inline std::string IntegerToString(const _Ty _Val) noexcept {// convert _Val to std::string
    static_assert(std::is_integral_v<_Ty>, "_Ty must be integral");
    using _UTy = std::make_unsigned_t<_Ty>;
    char _Buff[21];// can hold -2^63 and 2^64 - 1, plus NUL
    char *const _Buff_end = std::end(_Buff);
    char *_RNext = _Buff_end;
    const auto _UVal = static_cast<_UTy>(_Val);
    if (_Val < 0) {
        _RNext = UIntegral_to_buff(_RNext, static_cast<_UTy>(0 - _UVal));
        *--_RNext = '-';
    } else {
        _RNext = UIntegral_to_buff(_RNext, _UVal);
    }

    return std::string(_RNext, _Buff_end);
}
template<class _Ty>
inline void IntegerToString(const _Ty _Val, std::string &str) noexcept {// convert _Val to std::string
    static_assert(std::is_integral_v<_Ty>, "_Ty must be integral");
    using _UTy = std::make_unsigned_t<_Ty>;
    char _Buff[21];// can hold -2^63 and 2^64 - 1, plus NUL
    char *const _Buff_end = std::end(_Buff);
    char *_RNext = _Buff_end;
    const auto _UVal = static_cast<_UTy>(_Val);
    if (_Val < 0) {
        _RNext = UIntegral_to_buff(_RNext, static_cast<_UTy>(0 - _UVal));
        *--_RNext = '-';
    } else {
        _RNext = UIntegral_to_buff(_RNext, _UVal);
    }
    str.append(_RNext, _Buff_end - _RNext);
}
inline std::string to_string(double _Val) noexcept {
    const auto _Len = static_cast<size_t>(_CSTD _scprintf("%f", _Val));
    std::string _Str(_Len, '\0');
    _CSTD sprintf_s(&_Str[0], _Len + 1, "%f", _Val);
    return _Str;
}
inline void to_string(double _Val, std::string &str) noexcept {
    const auto _Len = static_cast<size_t>(_CSTD _scprintf("%f", _Val));
    size_t oldSize = str.size();
    str.resize(oldSize + _Len);
    _CSTD sprintf_s(&str[oldSize], _Len + 1, "%f", _Val);
}
inline std::string to_string(float _Val) noexcept {
    return to_string((double)_Val);
}

inline std::string to_string(int32_t _Val) noexcept {
    return IntegerToString(_Val);
}
inline std::string to_string(uint32_t _Val) noexcept {
    return IntegerToString(_Val);
}
inline std::string to_string(int16_t _Val) noexcept {
    return IntegerToString(_Val);
}
inline std::string to_string(uint16_t _Val) noexcept {
    return IntegerToString(_Val);
}
inline std::string to_string(int8_t _Val) noexcept {
    return IntegerToString(_Val);
}
inline std::string to_string(uint8_t _Val) noexcept {
    return IntegerToString(_Val);
}
inline std::string to_string(int64_t _Val) noexcept {
    return IntegerToString(_Val);
}
inline std::string to_string(uint64_t _Val) noexcept {
    return IntegerToString(_Val);
}

inline void to_string(float _Val, std::string &str) noexcept {
    to_string((double)_Val, str);
}

inline void to_string(int32_t _Val, std::string &str) noexcept {
    IntegerToString(_Val, str);
}
inline void to_string(uint32_t _Val, std::string &str) noexcept {
    IntegerToString(_Val, str);
}
inline void to_string(int16_t _Val, std::string &str) noexcept {
    IntegerToString(_Val, str);
}
inline void to_string(uint16_t _Val, std::string &str) noexcept {
    IntegerToString(_Val, str);
}
inline void to_string(int8_t _Val, std::string &str) noexcept {
    IntegerToString(_Val, str);
}
inline void to_string(uint8_t _Val, std::string &str) noexcept {
    IntegerToString(_Val, str);
}
inline void to_string(int64_t _Val, std::string &str) noexcept {
    IntegerToString(_Val, str);
}
inline void to_string(uint64_t _Val, std::string &str) noexcept {
    IntegerToString(_Val, str);
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
