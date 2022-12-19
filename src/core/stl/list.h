#pragma once

#include <EASTL/list.h>
#include <EASTL/slist.h>
#include <EASTL/fixed_list.h>
#include <EASTL/fixed_slist.h>

namespace luisa {

template<typename T>
using forward_list = eastl::slist<T>;

template<typename T>
using list = eastl::list<T>;

template<typename T, size_t node_count, bool allow_overflow = true>
using fixed_forward_list = eastl::fixed_slist<T, node_count, allow_overflow>;

template<typename T, size_t node_count, bool allow_overflow = true>
using fixed_list = eastl::fixed_list<T, node_count, allow_overflow>;

}// namespace luisa
