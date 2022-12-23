#pragma once
#include <core/stl/vector.h>
#include <core/stl/memory.h>
#include <core/stl/functional.h>
#include <core/stl/hash.h>
#include <core/stl/unordered_dense.h>

namespace luisa {
template<typename K, typename V,
         typename Hash = hash<K>,
         typename Eq = std::equal_to<>,
         typename Allocator = luisa::allocator<std::pair<K, V>>,
         typename Vector = vector<std::pair<K, V>>>
using unordered_map = ankerl::unordered_dense::map<K, V, Hash, Eq, Allocator, Vector>;

template<typename K,
         typename Hash = hash<K>,
         typename Eq = std::equal_to<>,
         typename Allocator = luisa::allocator<K>,
         typename Vector = vector<K>>
using unordered_set = ankerl::unordered_dense::set<K, Hash, Eq, Allocator, Vector>;

}// namespace luisa
