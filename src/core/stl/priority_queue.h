#pragma once

#include <EASTL/priority_queue.h>

#include <core/stl/vector.h>
#include <core/stl/functional.h>

namespace luisa {

template<typename T,
         typename Container = vector<T>,
         typename Compare = less<>>
using priority_queue = eastl::priority_queue<T, Container, Compare>;

}
