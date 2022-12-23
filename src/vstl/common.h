#pragma once

#include <vstl/config.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <condition_variable>
#include <vstl/log.h>
#include <vstl/spin_mutex.h>
#include <vstl/unique_ptr.h>
#include <vstl/hash_map.h>
#include <vstl/vstring.h>
#include <string_view>
#include <vstl/array.h>
#include <core/stl/hash.h>
#include <core/stl/unordered_dense.h>

namespace vstd {
using string_view = std::basic_string_view<char>;
using wstring_view = std::basic_string_view<wchar_t>;
template<typename K, typename V,
         typename Hash = luisa::hash<K>,
         typename Eq = std::equal_to<>,
         typename Allocator = luisa::allocator<std::pair<K, V>>,
         typename Vector = vstd::vector<std::pair<K, V>>>
using unordered_map = ankerl::unordered_dense::map<K, V, Hash, Eq, Allocator, Vector>;

template<typename K,
         typename Hash = luisa::hash<K>,
         typename Eq = std::equal_to<>,
         typename Allocator = luisa::allocator<K>,
         typename Vector = vstd::vector<K>>
using unordered_set = ankerl::unordered_dense::set<K, Hash, Eq, Allocator, Vector>;
}// namespace vstd
inline constexpr vstd::string_view operator"" _sv(const char *_Str, size_t _Len) noexcept {
    return vstd::string_view(_Str, _Len);
}

inline constexpr vstd::wstring_view operator"" _sv(const wchar_t *_Str, size_t _Len) noexcept {
    return vstd::wstring_view(_Str, _Len);
}
