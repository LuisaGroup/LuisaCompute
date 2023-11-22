#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"

namespace luisa::shader {

[[expr("dispatch_id")]] extern uint3 dispatch_id();

template <floatN T> 
[[callop("SIN")]] extern T sin(T rad);
template <floatN T> 
[[callop("COS")]] extern T cos(T rad);

template <floatN T> 
[[callop("ASIN")]] extern T asin(T x);

}// namespace luisa::shader