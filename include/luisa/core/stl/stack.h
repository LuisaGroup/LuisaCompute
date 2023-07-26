#pragma once

#include <EASTL/stack.h>
#include <luisa/core/stl/vector.h>

namespace luisa {

template<typename T, typename Container = luisa::vector<T>>
using stack = eastl::stack<T, Container>;

}

