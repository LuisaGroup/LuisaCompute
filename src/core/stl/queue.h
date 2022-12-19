#pragma once

#include <EASTL/queue.h>

#include <core/stl/deque.h>

namespace luisa {

template<typename T, typename Container = luisa::deque<T>>
using queue = eastl::queue<T, Container>;

}// namespace luisa
