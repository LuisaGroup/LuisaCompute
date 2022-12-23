#pragma once

#include <EASTL/vector.h>
#include <EASTL/fixed_vector.h>
#include <EASTL/bitvector.h>

namespace luisa {

template<typename T>
using vector = eastl::vector<T>;

template<typename T, size_t node_count, bool allow_overflow = true>
using fixed_vector = eastl::fixed_vector<T, node_count, allow_overflow, eastl::allocator>;

using bitvector = eastl::bitvector<>;

}// namespace luisa
