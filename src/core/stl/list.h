#pragma once

#include <EASTL/list.h>
#include <EASTL/slist.h>
#include <core/stl/memory.h>

namespace luisa {

template<typename T>
using forward_list = eastl::slist<T>;

template<typename T>
using list = eastl::list<T>;

}// namespace luisa
