#pragma once
#include <luisa/resources/buffer.hpp>
#include <luisa/functions/math.hpp>

#include <std/type_traits>

namespace luisa::shader {
template<typename T>
	requires(std::is_same_v<T, float> || std::is_same_v<float, element_of<T>>)
static typename copy_dim<uint, T>::type float_pack_to_uint(T val) {
	using RetType = typename copy_dim<uint, T>::type;
	RetType uvalue = bit_cast<RetType>(val);
	return ite(uvalue >> RetType(31u) == RetType(0u), uvalue | RetType(1u << 31u), ~uvalue);
}

template<typename T>
	requires(std::is_same_v<T, uint> || std::is_same_v<uint, element_of<T>>)
static typename copy_dim<float, T>::type uint_unpack_to_float(T val) {
	using RetType = typename copy_dim<uint, T>::type;
	RetType uvalue = ite(val >> RetType(31u) == 0, ~val, val & RetType(~(1u << 31u)));
	return bit_cast<typename copy_dim<float, T>::type>(uvalue);
}

static float float_atomic_min(Buffer<uint>& buffer, uint index, float value) {
	return uint_unpack_to_float(buffer.atomic_fetch_min(index, float_pack_to_uint(value)));
}
static float float_atomic_max(Buffer<uint>& buffer, uint index, float value) {
	return uint_unpack_to_float(buffer.atomic_fetch_max(index, float_pack_to_uint(value)));
}
static float float_atomic_add(
	Buffer<uint>& buffer,
	uint index,
	float value) {
	uint old = buffer.load(index);
	while (true) {
		uint r = buffer.atomic_compare_exchange(index, old, bit_cast<uint>(bit_cast<float>(old) + value));
		if (r == old) {
			break;
		}
		old = r;
	}
	return bit_cast<float>(old);
}

}// namespace luisa::shader