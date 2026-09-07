#pragma once

#include <stack>
#include <luisa/core/stl/vector.h>

namespace luisa {

template<typename T, typename Container = luisa::vector<T>>
using stack = std::stack<T, Container>;

}// namespace luisa
