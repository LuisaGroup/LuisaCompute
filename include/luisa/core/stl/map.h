#pragma once

#include <EASTL/map.h>
#include <EASTL/set.h>

namespace luisa {

template<typename Key,
         typename Compare = std::less<>,
         typename allocator = luisa::allocator<Key>>
using set = eastl::set<Key, Compare, allocator>;

template<typename Key, typename Value,
         typename Compare = std::less<>,
         typename allocator = luisa::allocator<std::pair<const Key, Value>>>
using map = eastl::map<Key, Value, Compare, allocator>;

template<typename Key,
         typename Compare = std::less<>,
         typename allocator = luisa::allocator<Key>>
using multiset = eastl::multiset<Key, Compare, allocator>;

template<typename Key, typename Value,
         typename Compare = std::less<>,
         typename allocator = luisa::allocator<std::pair<const Key, Value>>>
using multimap = eastl::multimap<Key, Value, Compare, allocator>;

}// namespace luisa
