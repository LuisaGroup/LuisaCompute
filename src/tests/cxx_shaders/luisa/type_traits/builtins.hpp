#pragma once
#include "./../attributes.hpp"

namespace luisa::shader {

using int16 = short;
using uint16 = unsigned short;
using int32 = int;
using int64 = long long;
using uint32 = unsigned int;
using uint = uint32;
using uint64 = unsigned long long;
#pragma clang diagnostic ignored "-Winvalid-offsetof"
#define _offsetof(Type, member) uint(__builtin_offsetof(Type, member))

struct [[builtin("half")]] half {
	[[ignore]] half() = default;
	[[ignore]] half(float);
	[[ignore]] half(uint32);
	[[ignore]] half(int32);
	[[cast]] operator float() const;
	[[cast]] operator uint() const;
	[[cast]] operator int() const;

private:
	short v;
};

}// namespace luisa::shader