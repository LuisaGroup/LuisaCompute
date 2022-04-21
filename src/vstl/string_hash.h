#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <vstl/MetaLib.h>
#include <iostream>
#include <string_view>
using namespace std::literals;

namespace vstd {
template<>
struct hash<std::string_view> {
    inline size_t operator()(const std::string_view &str) const noexcept {
        return Hash::CharArrayHash(str.data(), str.size());
    }
};

template<>
struct hash<char const *> {
    inline size_t operator()(char const *ptr) const noexcept {
        auto sz = strlen(ptr);
        return Hash::CharArrayHash(ptr, sz);
    }
};
template<>
struct hash<char *> {
    inline size_t operator()(char *ptr) const noexcept {
        auto sz = strlen(ptr);
        return Hash::CharArrayHash(ptr, sz);
    }
};
template<size_t i>
struct hash<char[i]> {
    inline size_t operator()(char const *ptr) const noexcept {
        return Hash::CharArrayHash(ptr, i - 1);
    }
};
template<>
struct hash<wchar_t const *> {
    static size_t wstrlen(wchar_t const *ptr) {
        size_t i = 0;
        while (*ptr != 0) {
            i += 2;
            ptr++;
        }
        return i;
    }
    inline size_t operator()(wchar_t const *ptr) const noexcept {
        auto sz = wstrlen(ptr);
        return Hash::CharArrayHash(reinterpret_cast<char const *>(ptr), sz);
    }
};
template<>
struct hash<wchar_t *> {
    inline size_t operator()(wchar_t *ptr) const noexcept {
        hash<wchar_t const *> hs;
        return hs(ptr);
    }
};
template<size_t i>
struct hash<wchar_t[i]> {
    inline size_t operator()(wchar_t const *ptr) const noexcept {
        return Hash::CharArrayHash(reinterpret_cast<char const *>(ptr), (i - 1) * 2);
    }
};

}// namespace vstd
