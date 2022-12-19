//
// Created by Mike Smith on 2022/12/20.
//

#pragma once

#include <EASTL/stack.h>
#include <core/stl/vector.h>

namespace luisa {

template<typename T, typename Container = luisa::vector<T>>
using stack = eastl::stack<T, Container>;

}
