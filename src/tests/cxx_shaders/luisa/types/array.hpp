#pragma once
#include "../attributes.hpp"
#include "../type_traits.hpp"

#include <std/type_traits>
#include <std/array>

namespace luisa::shader {

template<typename Type, uint32 size>
struct [[builtin("shared_array")]] SharedArray {
	static constexpr uint32 N = size;

	static constexpr bool has_atomics = std::is_same_v<Type, int32> || std::is_same_v<Type, uint32>;

	template<typename... Args>
	[[noignore]] constexpr SharedArray(Args... args) {
		set<0>(args...);
	}
	constexpr SharedArray() = default;

	template<uint32 start, typename... Args>
	[[noignore]] constexpr void set(Type v, Args... args) {
		set(start, v);
		set<start + 1>(args...);
	}
	template<uint32 start>
	[[noignore]] constexpr void set(Type v) { set(start, v); }
	[[noignore]] constexpr void set(uint32 loc, Type v) { access_(loc) = v; }
	[[access]] constexpr Type& access_(uint32 loc) { return v[loc]; }
	[[access]] constexpr Type& operator[](uint32 loc) { return v[loc]; }

	[[noignore]] constexpr Type get(uint32 loc) const { return access_(loc); }
	[[access]] constexpr Type access_(uint32 loc) const { return v[loc]; }
	[[access]] constexpr Type operator[](uint32 loc) const { return v[loc]; }

	[[callop("ATOMIC_EXCHANGE")]] Type atomic_exchange(uint32 loc, int32 desired)
		requires(has_atomics);
	[[callop("ATOMIC_COMPARE_EXCHANGE")]] Type atomic_compare_exchange(uint32 loc, int32 expected, int32 desired)
		requires(has_atomics);
	[[callop("ATOMIC_FETCH_ADD")]] Type atomic_fetch_add(uint32 loc, int32 val)
		requires(has_atomics);
	[[callop("ATOMIC_FETCH_SUB")]] Type atomic_fetch_sub(uint32 loc, int32 val)
		requires(has_atomics);
	[[callop("ATOMIC_FETCH_AND")]] Type atomic_fetch_and(uint32 loc, int32 val)
		requires(has_atomics);
	[[callop("ATOMIC_FETCH_OR")]] Type atomic_fetch_or(uint32 loc, int32 val)
		requires(has_atomics);
	[[callop("ATOMIC_FETCH_XOR")]] Type atomic_fetch_xor(uint32 loc, int32 val)
		requires(has_atomics);
	[[callop("ATOMIC_FETCH_MIN")]] Type atomic_fetch_min(uint32 loc, int32 val)
		requires(has_atomics);
	[[callop("ATOMIC_FETCH_MAX")]] Type atomic_fetch_max(uint32 loc, int32 val)
		requires(has_atomics);

private:
	// DONT EDIT THIS FIELD LAYOUT
	Type v[size];
};
}// namespace luisa::shader