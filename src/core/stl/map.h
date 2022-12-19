#pragma once

#define LUISA_COMPUTE_USE_BTREE_MAP
#ifdef LUISA_COMPUTE_USE_BTREE_MAP
#include <parallel_hashmap/phmap.h>
#include <parallel_hashmap/btree.h>
#else
#include <map>
#include <set>
#endif

#include <core/stl/memory.h>

namespace luisa {

#ifdef LUISA_COMPUTE_USE_BTREE_MAP
template<typename Key,
         typename Compare = std::less<>>
using set = phmap::btree_set<
    Key, Compare, luisa::allocator<Key>>;

template<typename Key, typename Value,
         typename Compare = std::less<>>
using map = phmap::btree_map<
    Key, Value, Compare, luisa::allocator<std::pair<const Key, Value>>>;

template<typename Key,
         typename Compare = std::less<>>
using multiset = phmap::btree_multiset<
    Key, Compare, luisa::allocator<Key>>;

template<typename Key, typename Value,
         typename Compare = std::less<>>
using multimap = phmap::btree_multimap<
    Key, Value, Compare, luisa::allocator<std::pair<const Key, Value>>>;
#else
template<typename Key,
         typename Compare = std::less<>>
using set = std::set<
    Key, Compare, luisa::allocator<Key>>;

template<typename Key, typename Value,
         typename Compare = std::less<>>
using map = std::map<
    Key, Value, Compare, luisa::allocator<std::pair<const Key, Value>>>;

template<typename Key,
         typename Compare = std::less<>>
using multiset = std::multiset<
    Key, Compare, luisa::allocator<Key>>;

template<typename Key, typename Value,
         typename Compare = std::less<>>
using multimap = std::multimap<
    Key, Value, Compare, luisa::allocator<std::pair<const Key, Value>>>;
#endif

}// namespace luisa
