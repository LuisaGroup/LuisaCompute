#pragma once
#include <luisa/type_traits.hpp>
#include <luisa/functions/math.hpp>
namespace luisa::shader {
namespace float16_detail {
constexpr static int shift = 13;
constexpr static int shiftSign = 16;

constexpr static int infN = 0x7F800000; // flt32 infinity
constexpr static int maxN = 0x477FE000; // max flt16 normal as a flt32
constexpr static int minN = 0x38800000; // min flt16 normal as a flt32
constexpr static int signN = 0x80000000;// flt32 sign bit

constexpr static int infC = infN >> shift;
constexpr static int nanN = (infC + 1) << shift;// minimum flt16 nan as a flt32
constexpr static int maxC = maxN >> shift;
constexpr static int minC = minN >> shift;
constexpr static int signC = signN >> shiftSign;// flt16 sign bit

constexpr static int mulN = 0x52000000;// (1 << 23) / minN
constexpr static int mulC = 0x33800000;// minN / (1 << (23 - shift))

constexpr static int subC = 0x003FF;// max flt32 subnormal down shifted
constexpr static int norC = 0x00400;// min flt32 normal down shifted

constexpr static int maxD = infC - maxC - 1;
constexpr static int minD = minC - subC - 1;
}// namespace float16_detail

static uint32 float16_compress(float value) {
	using namespace float16_detail;
	if (value == 0)
		return 0u;
	float v_f;
	int32 v_si;

	float s_f;
	int32 s_si;

	v_f = value;
	v_si = bit_cast<int>(v_f);
	uint32 sign = v_si & signN;
	v_si ^= sign;
	sign >>= shiftSign;// logical shift
	s_si = mulN;
	v_f = bit_cast<float>(v_si);
	s_f = bit_cast<float>(s_si);
	s_si = (int)(s_f * v_f);// correct subnormals
	v_si ^= (s_si ^ v_si) & -(minN > v_si);
	v_si ^= (infN ^ v_si) & -((infN > v_si) & (v_si > maxN));
	v_si ^= (nanN ^ v_si) & -((nanN > v_si) & (v_si > infN));
	v_si = (int)(((uint32)v_si) >> shift);// logical shift
	v_si ^= ((v_si - maxD) ^ v_si) & -(v_si > maxC);
	v_si ^= ((v_si - minD) ^ v_si) & -(v_si > subC);
	return ((uint32)v_si | sign);
}

static float float16_decompress(uint value) {
	using namespace float16_detail;
	if (value == 0u) return 0.f;
	int32 v_si;

	float s_f;
	int32 s_si;

	v_si = (int)value;
	int32 sign = v_si & signC;
	v_si ^= sign;
	sign <<= shiftSign;
	v_si ^= ((v_si + minD) ^ v_si) & -(v_si > subC);
	v_si ^= ((v_si + maxD) ^ v_si) & -(v_si > maxC);
	s_si = mulC;
	s_f = bit_cast<float>(s_si);
	s_f *= v_si;
	int32 mask = -(norC > v_si);
	v_si <<= shift;
	v_si ^= (s_si ^ v_si) & mask;
	v_si |= sign;
	return bit_cast<float>(v_si);
}

}// namespace luisa::shader