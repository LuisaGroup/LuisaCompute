#pragma once

#include <deque>
#include <luisa/core/stl/memory.h>

namespace luisa {

template<typename T>
using deque = std::deque<T, luisa::allocator<T>>;

}// namespace luisa
