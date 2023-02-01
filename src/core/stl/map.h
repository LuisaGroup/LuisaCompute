#pragma once

#define LUISA_COMPUTE_USE_BTREE_MAP
#ifdef LUISA_COMPUTE_USE_BTREE_MAP
#include <parallel_hashmap/btree.h>
#else
#include <map>
#include <set>
#endif

#include <core/stl/memory.h>

namespace luisa {

#ifdef LUISA_COMPUTE_USE_BTREE_MAP
template<typename Key,
         typename Compare = std::less<>, typename allocator = luisa::allocator<Key>>
using set = phmap::btree_set<
    Key, Compare, allocator>;

template<typename Key, typename Value,
         typename Compare = std::less<>, typename allocator = luisa::allocator<std::pair<const Key, Value>>>
using map = phmap::btree_map<
    Key, Value, Compare, allocator>;

template<typename Key,
         typename Compare = std::less<>, typename allocator = luisa::allocator<Key>>
using multiset = phmap::btree_multiset<
    Key, Compare, allocator>;

template<typename Key, typename Value,
         typename Compare = std::less<>, typename allocator = luisa::allocator<std::pair<const Key, Value>>>
using multimap = phmap::btree_multimap<
    Key, Value, Compare, allocator>;
#else
template<typename Key,
         typename Compare = std::less<>, typename allocator = luisa::allocator<Key>>
using set = std::set<
    Key, Compare, allocator>;

template<typename Key, typename Value,
         typename Compare = std::less<>, typename allocator = luisa::allocator<std::pair<const Key, Value>>>
using map = std::map<
    Key, Value, Compare, allocator>;

template<typename Key,
         typename Compare = std::less<>, typename allocator = luisa::allocator<Key>>
using multiset = std::multiset<
    Key, Compare, allocator>;

template<typename Key, typename Value,
         typename Compare = std::less<>, typename allocator = luisa::allocator<std::pair<const Key, Value>>>
using multimap = std::multimap<
    Key, Value, Compare, allocator>;
#endif

}// namespace luisa
