#pragma once
#include "attributes.hpp"
#include "type_traits.hpp"
#include "numeric.hpp"

#include "types.hpp"
#include "resources.hpp"

#include "functions.hpp"
#include "raytracing.hpp"

#include <std/type_traits>

namespace luisa::shader {

template<typename Resource, typename T>
static void store_2d(Resource& r, uint32 row_pitch, uint2 pos, T val) {
	using ResourceType = std::remove_cvref_t<Resource>;
	if constexpr (std::is_same_v<ResourceType, Buffer<T>>)
		r.store(pos.x + pos.y * row_pitch, val);
	else if constexpr (std::is_same_v<ResourceType, Image<scalar_type<T>>>)
		r.store(pos, val);
}
template<concepts::primitive T>
constexpr void swap(T& l, T& r) {
	T tmp = l;
	l = r;
	r = tmp;
}
}// namespace luisa::shader
