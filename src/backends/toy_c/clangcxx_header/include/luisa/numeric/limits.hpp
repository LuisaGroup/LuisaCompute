#pragma once
#include <luisa/type_traits.hpp>
namespace luisa::shader {
constexpr static float min_float32 =  -0x1.fffffe0000000p+127f;
constexpr static float max_float32 = 0x1.fffffe0000000p+127f;
constexpr static double min_float64 = -0x1.fffffffffffffp+1023;
constexpr static double max_float64 = 0x1.fffffffffffffp+1023;
constexpr static int16 max_int16 = 32767;
constexpr static int16 min_int16 = -max_int16 - 1;
constexpr static int32 max_int32 = 2147483647;
constexpr static int32 min_int32 = -max_int32 - 1;
constexpr static int64 max_int64 = 9223372036854775807ull;
constexpr static int64 min_int64 = -max_int64 - 1;
constexpr static uint16 max_uint16 = 65535u;
constexpr static uint16 min_uint16 = 0;
constexpr static uint32 max_uint32 = 4294967295u;
constexpr static uint32 min_uint32 = 0;
constexpr static uint64 max_uint64 = 18446744073709551615ull;
constexpr static uint64 min_uint64 = 0;
}// namespace luisa::shader