#pragma once

#include <EASTL/vector_map.h>
#include <EASTL/vector_set.h>
#include <EASTL/vector_multimap.h>
#include <EASTL/vector_multiset.h>

#include <core/stl/functional.h>

namespace luisa {

template<typename Key, typename Value,
         typename Compare = equal_to<>>
using vector_map = eastl::vector_map<Key, Value, Compare>;

template<typename Key, typename Value,
         typename Compare = equal_to<>>
using vector_multimap = eastl::vector_multimap<Key, Value, Compare>;

template<typename Key,
         typename Compare = equal_to<>>
using vector_set = eastl::vector_set<Key, Compare>;

template<typename Key,
         typename Compare = equal_to<>>
using vector_multiset = eastl::vector_multiset<Key, Compare>;

}// namespace luisa
