#pragma once

#include <queue>
#include <functional>
#include <luisa/core/stl/vector.h>

namespace luisa {

template<typename T,
         typename Container = luisa::vector<T>,
         typename Compare = std::less<>>
using priority_queue = std::priority_queue<T, Container, Compare>;

}// namespace luisa
