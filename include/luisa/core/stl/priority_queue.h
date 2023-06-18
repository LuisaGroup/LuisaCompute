#pragma once

#include <EASTL/priority_queue.h>

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/functional.h>

namespace luisa {

template<typename T,
         typename Container = vector<T>,
         typename Compare = less<>>
using priority_queue = eastl::priority_queue<T, Container, Compare>;

}

