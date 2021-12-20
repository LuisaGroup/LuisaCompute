#pragma once

#include <vstl/config.h>
#include <memory>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cassert>
#include <condition_variable>
#include <vstl/Log.h>
#include <vstl/spin_mutex.h>
#include <vstl/unique_ptr.h>
#include <vstl/HashMap.h>
#include <vstl/vstring.h>
#include <core/mathematics.h>
namespace vstd {
template<
    class T,
    std::size_t Extent = std::dynamic_extent>
using span = std::span<T, Extent>;
using string_view = std::basic_string_view<char>;
using wstring_view = std::basic_string_view<wchar_t>;
}// namespace vstd
inline constexpr vstd::string_view operator"" _sv(const char *_Str, size_t _Len) noexcept {
    return vstd::string_view(_Str, _Len);
}

inline constexpr vstd::wstring_view operator"" _sv(const wchar_t *_Str, size_t _Len) noexcept {
    return vstd::wstring_view(_Str, _Len);
}
