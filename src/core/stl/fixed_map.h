#pragma once

#include <EASTL/fixed_map.h>
#include <EASTL/fixed_set.h>
#include <EASTL/fixed_hash_map.h>
#include <EASTL/fixed_hash_set.h>

#include <core/stl/hash.h>
#include <core/stl/functional.h>

namespace luisa {

template<typename Key, typename Value,
         size_t node_count, bool allow_overflow = true,
         typename Hash = hash<Key>, typename Eq = equal_to<>>
using fixed_unordered_map = eastl::fixed_hash_map<
    Key, Value, node_count, node_count + 1u, allow_overflow, Hash, Eq>;

template<typename Key,
         size_t node_count, bool allow_overflow = true,
         typename Hash = hash<Key>, typename Eq = equal_to<>>
using fixed_unordered_set = eastl::fixed_hash_set<
    Key, node_count, node_count + 1u, allow_overflow, Hash, Eq>;

template<typename Key, typename Value,
         size_t node_count, bool allow_overflow = true,
         typename Compare = less<>>
using fixed_map = eastl::fixed_map<
    Key, Value, node_count, allow_overflow, Compare>;

template<typename Key,
         size_t node_count, bool allow_overflow = true,
         typename Compare = less<>>
using fixed_set = eastl::fixed_set<
    Key, node_count, allow_overflow, Compare>;

template<typename Key, typename Value,
         size_t node_count, bool allow_overflow = true,
         typename Compare = less<>>
using fixed_multimap = eastl::fixed_multimap<
    Key, Value, node_count, allow_overflow, Compare>;

template<typename Key,
         size_t node_count, bool allow_overflow = true,
         typename Compare = less<>>
using fixed_multiset = eastl::fixed_multiset<
    Key, node_count, allow_overflow, Compare>;

}// namespace luisa
