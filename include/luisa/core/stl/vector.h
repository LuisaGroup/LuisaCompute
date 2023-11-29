#pragma once

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include <EASTL/vector.h>
#include <EASTL/fixed_vector.h>
#include <EASTL/bitvector.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace luisa {

template<typename T>
using vector = eastl::vector<T>;

template<typename T, size_t node_count, bool allow_overflow = true>
using fixed_vector = eastl::fixed_vector<T, node_count, allow_overflow>;

using bitvector = eastl::bitvector<>;

}// namespace luisa
