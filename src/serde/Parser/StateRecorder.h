#pragma once

#include <bitset>
#include <vstl/Common.h>

namespace toolhub::db::parser {
template<typename T>
struct StateRecorder {
	T states[std::numeric_limits<char>::max() + 1];
	template<typename Func>
	StateRecorder(
		Func&& initFunc) {
		if constexpr (std::is_trivially_constructible_v<T>)
			memset(states, 0, sizeof(states));
		initFunc(states);
	}
	T const& Get(char p) const {
		return states[p];
	}
	T const& operator[](char p) const {
		return states[p];
	}
};

template<>
struct StateRecorder<bool> {
	using ArrayType = std::bitset<std::numeric_limits<char>::max() + 1>;
	ArrayType states;
	template<typename Func>
	StateRecorder(
		Func&& initFunc) {
		initFunc(states);
	}
	bool Get(char p) const {
		return states[p];
	}
	bool operator[](char p) const {
		return states[p];
	}
};

}// namespace toolhub::db::parser