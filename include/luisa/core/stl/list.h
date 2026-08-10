#pragma once

#include <list>
#include <forward_list>
#include <luisa/core/stl/memory.h>

namespace luisa {

template<typename T>
using forward_list = std::forward_list<T, luisa::allocator<T>>;

template<typename T>
using list = std::list<T, luisa::allocator<T>>;

template<typename T, size_t node_count, bool allow_overflow = true>
using fixed_forward_list = std::forward_list<T, luisa::allocator<T>>;

template<typename T, size_t node_count, bool allow_overflow = true>
using fixed_list = std::list<T, luisa::allocator<T>>;

}// namespace luisa
