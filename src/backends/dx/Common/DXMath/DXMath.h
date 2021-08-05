#pragma once
#include <util/vstlconfig.h>
#include <Windows.h>
#include <DirectXMath.h>
#include <intrin.h>

typedef DirectX::XMFLOAT2 float2;
typedef DirectX::XMFLOAT3 float3;
typedef DirectX::XMFLOAT4 float4;
typedef DirectX::XMUINT2 uint2;
typedef DirectX::XMUINT3 uint3;
typedef DirectX::XMUINT4 uint4;
typedef DirectX::XMINT2 int2;
typedef DirectX::XMINT3 int3;
typedef DirectX::XMINT4 int4;
typedef uint32_t uint;
typedef DirectX::XMFLOAT4X4 float4x4;
typedef DirectX::XMFLOAT3X3 float3x3;
typedef DirectX::XMFLOAT3X4 float3x4;
typedef DirectX::XMFLOAT4X3 float4x3;

#include <util/Hash.h>
#include <Common/DXMath/Vector.h>
#include <Common/DXMath/Quaternion.h>
#include <Common/DXMath/Matrix4.h>
#include <Common/DXMath/Matrix3.h>
#include <Common/DynamicDLL.h>
#include <memory>
#include <initializer_list>

using namespace DirectX;
inline float XM_CALLCONV CombineVector4(XMVECTOR const& V2) noexcept {
#ifdef _XM_NO_INTRINSICS_
	return V2.vector4_f32[0] + V2.vector4_f32[1] + V2.vector4_f32[2] + V2.vector4_f32[3];
#else
	XMVECTOR vTemp2 = _mm_shuffle_ps(V2, V2, _MM_SHUFFLE(1, 0, 0, 0));// Copy X to the Z position and Y to the W position
	vTemp2 = _mm_add_ps(vTemp2, V2);								  // Add Z = X+Z; W = Y+W;
	return vTemp2.m128_f32[2] + vTemp2.m128_f32[3];
#endif
}

inline float XM_CALLCONV dot(const Math::Vector4& vec, const Math::Vector4& vec1) noexcept {
	return CombineVector4(vec * vec1);
}

inline void XM_CALLCONV Float4x4ToFloat4x3(const float4x4& f, float4x3& result) noexcept {
	memcpy(&result._11, &f._11, sizeof(float3));
	memcpy(&result._21, &f._21, sizeof(float3));
	memcpy(&result._31, &f._31, sizeof(float3));
	memcpy(&result._41, &f._41, sizeof(float3));
}

inline void XM_CALLCONV Float4x3ToFloat4x4(const float4x3& f, float4x4& result) noexcept {
	memcpy(&result._11, &f._11, sizeof(float3));
	memcpy(&result._21, &f._21, sizeof(float3));
	memcpy(&result._31, &f._31, sizeof(float3));
	memcpy(&result._41, &f._41, sizeof(float3));
	result._14 = 0;
	result._24 = 0;
	result._34 = 0;
	result._44 = 1;
}

inline Math::Vector4 XM_CALLCONV abs(const Math::Vector4& vec) noexcept {
	XMVECTOR const& V = (XMVECTOR const&)vec;
#if defined(_XM_NO_INTRINSICS_)
	XMVECTORF32 vResult = {{{fabsf(V.vector4_f32[0]),
							 fabsf(V.vector4_f32[1]),
							 fabsf(V.vector4_f32[2]),
							 fabsf(V.vector4_f32[3])}}};
	return vResult.v;
#elif defined(_XM_ARM_NEON_INTRINSICS_)
	return vabsq_f32(V);
#elif defined(_XM_SSE_INTRINSICS_)
	XMVECTOR vResult = _mm_setzero_ps();
	vResult = _mm_sub_ps(vResult, V);
	vResult = _mm_max_ps(vResult, V);
	return vResult;
#endif
}

inline Math::Vector3 XM_CALLCONV abs(const Math::Vector3& vec) noexcept {
	XMVECTOR const& V = (XMVECTOR const&)vec;
#if defined(_XM_NO_INTRINSICS_)
	XMVECTORF32 vResult = {{{fabsf(V.vector4_f32[0]),
							 fabsf(V.vector4_f32[1]),
							 fabsf(V.vector4_f32[2]),
							 fabsf(V.vector4_f32[3])}}};
	return vResult.v;
#elif defined(_XM_ARM_NEON_INTRINSICS_)
	return vabsq_f32(V);
#elif defined(_XM_SSE_INTRINSICS_)
	XMVECTOR vResult = _mm_setzero_ps();
	vResult = _mm_sub_ps(vResult, V);
	vResult = _mm_max_ps(vResult, V);
	return vResult;
#endif
}

inline Math::Vector4 XM_CALLCONV clamp(
	const Math::Vector4& v,
	const Math::Vector4& min,
	const Math::Vector4& max) noexcept {
	FXMVECTOR& V = (FXMVECTOR&)v;
	FXMVECTOR& Min = (FXMVECTOR&)min;
	FXMVECTOR& Max = (FXMVECTOR&)max;
#if defined(_XM_NO_INTRINSICS_)

	XMVECTOR Result;
	Result = XMVectorMax(Min, V);
	Result = XMVectorMin(Max, Result);
	return Result;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMVECTOR vResult;
	vResult = vmaxq_f32(Min, V);
	vResult = vminq_f32(Max, vResult);
	return vResult;
#elif defined(_XM_SSE_INTRINSICS_)
	XMVECTOR vResult;
	vResult = _mm_max_ps(Min, V);
	vResult = _mm_min_ps(Max, vResult);
	return vResult;
#endif
}

inline Math::Vector3 XM_CALLCONV clamp(
	const Math::Vector3& v,
	const Math::Vector3& min,
	const Math::Vector3& max) noexcept {
	FXMVECTOR& V = (FXMVECTOR&)v;
	FXMVECTOR& Min = (FXMVECTOR&)min;
	FXMVECTOR& Max = (FXMVECTOR&)max;
#if defined(_XM_NO_INTRINSICS_)

	XMVECTOR Result;
	Result = XMVectorMax(Min, V);
	Result = XMVectorMin(Max, Result);
	return Result;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMVECTOR vResult;
	vResult = vmaxq_f32(Min, V);
	vResult = vminq_f32(Max, vResult);
	return vResult;
#elif defined(_XM_SSE_INTRINSICS_)
	XMVECTOR vResult;
	vResult = _mm_max_ps(Min, V);
	vResult = _mm_min_ps(Max, vResult);
	return vResult;
#endif
}

inline Math::Vector4 XM_CALLCONV lerp(const Math::Vector4& aa, const Math::Vector4& bb, float t) noexcept {
	XMVECTOR const& V0 = (XMVECTOR const&)aa;
	XMVECTOR const& V1 = (XMVECTOR const&)bb;
#if defined(_XM_NO_INTRINSICS_)

	XMVECTOR Scale = XMVectorReplicate(t);
	XMVECTOR Length = XMVectorSubtract(V1, V0);
	return XMVectorMultiplyAdd(Length, Scale, V0);

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMVECTOR L = vsubq_f32(V1, V0);
	return vmlaq_n_f32(V0, L, t);
#elif defined(_XM_SSE_INTRINSICS_)
	XMVECTOR L = _mm_sub_ps(V1, V0);
	XMVECTOR S = _mm_set_ps1(t);
	XMVECTOR Result = _mm_mul_ps(L, S);
	return _mm_add_ps(Result, V0);
#endif
}
inline Math::Vector3 XM_CALLCONV lerp(const Math::Vector3& aa, const Math::Vector3& bb, float t) noexcept {
	XMVECTOR const& V0 = (XMVECTOR const&)aa;
	XMVECTOR const& V1 = (XMVECTOR const&)bb;
#if defined(_XM_NO_INTRINSICS_)

	XMVECTOR Scale = XMVectorReplicate(t);
	XMVECTOR Length = XMVectorSubtract(V1, V0);
	return XMVectorMultiplyAdd(Length, Scale, V0);

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMVECTOR L = vsubq_f32(V1, V0);
	return vmlaq_n_f32(V0, L, t);
#elif defined(_XM_SSE_INTRINSICS_)
	XMVECTOR L = _mm_sub_ps(V1, V0);
	XMVECTOR S = _mm_set_ps1(t);
	XMVECTOR Result = _mm_mul_ps(L, S);
	return _mm_add_ps(Result, V0);
#endif
}

inline Math::Vector4 XM_CALLCONV lerp(const Math::Vector4& aa, const Math::Vector4& bb, const Math::Vector4& t) noexcept {
	XMVECTOR const& V0 = (XMVECTOR const&)aa;
	XMVECTOR const& V1 = (XMVECTOR const&)bb;
#if defined(_XM_NO_INTRINSICS_)

	XMVECTOR Scale = (XMVECTOR const&)t;
	XMVECTOR Length = XMVectorSubtract(V1, V0);
	return XMVectorMultiplyAdd(Length, Scale, V0);

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMVECTOR L = vsubq_f32(V1, V0);
	return vmlaq_n_f32(V0, L, t);
#elif defined(_XM_SSE_INTRINSICS_)
	XMVECTOR L = _mm_sub_ps(V1, V0);
	XMVECTOR const& S = (XMVECTOR const&)t;
	XMVECTOR Result = _mm_mul_ps(L, S);
	return _mm_add_ps(Result, V0);
#endif
}
inline float XM_CALLCONV lerp(float aa, float bb, float cc) noexcept {
	float l = bb - aa;
	return aa + l * cc;
}
inline float XM_CALLCONV lerpFloat(float aa, float bb, float cc) noexcept {
	float l = bb - aa;
	return aa + l * cc;
}
inline Math::Vector3 XM_CALLCONV lerp(const Math::Vector3& aa, const Math::Vector3& bb, const Math::Vector3& t) noexcept {
	XMVECTOR const& V0 = (XMVECTOR const&)aa;
	XMVECTOR const& V1 = (XMVECTOR const&)bb;
#if defined(_XM_NO_INTRINSICS_)

	XMVECTOR Scale = (XMVECTOR const&)t;
	XMVECTOR Length = XMVectorSubtract(V1, V0);
	return XMVectorMultiplyAdd(Length, Scale, V0);

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMVECTOR L = vsubq_f32(V1, V0);
	return vmlaq_n_f32(V0, L, t);
#elif defined(_XM_SSE_INTRINSICS_)
	XMVECTOR L = _mm_sub_ps(V1, V0);
	XMVECTOR const& S = (XMVECTOR const&)t;
	XMVECTOR Result = _mm_mul_ps(L, S);
	return _mm_add_ps(Result, V0);
#endif
}

inline float XM_CALLCONV CombineVector3(XMVECTOR const& v) noexcept {
	return v.m128_f32[0] + v.m128_f32[1] + v.m128_f32[2];
}

inline float XM_CALLCONV dot(const Math::Vector3& vec, const Math::Vector3& vec1) noexcept {
	return CombineVector3(vec * vec1);
}
inline int2 XM_CALLCONV mul(const int2& a, const int2& b) noexcept {
	return int2(a.x * b.x, a.y * b.y);
}

inline uint2 XM_CALLCONV mul(const uint2& a, const uint2& b) noexcept {
	return {a.x * b.x, a.y * b.y};
}

inline int3 XM_CALLCONV mul(const int3& a, const int3& b) noexcept {
	return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline uint3 XM_CALLCONV mul(const uint3& a, const uint3& b) noexcept {
	return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline int4 XM_CALLCONV mul(const int4& a, const int4& b) noexcept {
	return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

inline uint4 XM_CALLCONV mul(const uint4& a, const uint4& b) noexcept {
	return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

inline float2 XM_CALLCONV mul(const float2& a, const float2& b) noexcept {
	return {a.x * b.x, a.y * b.y};
}

inline float3 XM_CALLCONV mul(const float3& a, const float3& b) noexcept {
	return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline float4 XM_CALLCONV mul(const float4& a, const float4& b) noexcept {
	return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

inline Math::Vector4 XM_CALLCONV mul(const Math::Matrix4& m, const Math::Vector4& vec) noexcept {
	Math::Matrix4& mat = (Math::Matrix4&)m;
	return {
		dot(mat[0], vec),
		dot(mat[1], vec),
		dot(mat[2], vec),
		dot(mat[3], vec)};
}

inline Math::Vector3 XM_CALLCONV mul(const Math::Matrix3& m, const Math::Vector3& vec) noexcept {
	Math::Matrix4& mat = (Math::Matrix4&)m;
	return {
		CombineVector3(reinterpret_cast<Math::Vector3&>(mat[0]) * vec),
		CombineVector3(reinterpret_cast<Math::Vector3&>(mat[1]) * vec),
		CombineVector3(reinterpret_cast<Math::Vector3&>(mat[2]) * vec)};
}

inline Math::Vector3 XM_CALLCONV sqrt(const Math::Vector3& vec) noexcept {
#if defined(_XM_NO_INTRINSICS_)
	return Math::Vector3(XMVectorSqrt((FXMVECTOR&)vec));
#else
	return _mm_sqrt_ps((XMVECTOR const&)vec);
#endif
}

inline Math::Vector4 XM_CALLCONV sqrt(const Math::Vector4& vec) noexcept {
#if defined(_XM_NO_INTRINSICS_)
	return Math::Vector4(XMVectorSqrt((FXMVECTOR&)vec));
#else
	return _mm_sqrt_ps((XMVECTOR const&)vec);
#endif
}

inline float XM_CALLCONV length(const Math::Vector3& vec1) noexcept {
	Math::Vector3 diff = vec1 * vec1;
	float dotValue = CombineVector3((XMVECTOR const&)diff);
	return sqrt(dotValue);
}

inline float XM_CALLCONV lengthsq(const Math::Vector3& vec1) noexcept {
	Math::Vector3 diff = vec1 * vec1;
	return CombineVector3((XMVECTOR const&)diff);
}

inline float XM_CALLCONV lengthsq(const Math::Vector4& vec1) noexcept {
	Math::Vector4 diff = vec1 * vec1;
	return CombineVector4((XMVECTOR const&)diff);
}

inline float XM_CALLCONV distance(const Math::Vector3& vec1, const Math::Vector3& vec2) noexcept {
	Math::Vector3 diff = vec1 - vec2;
	diff *= diff;
	float dotValue = CombineVector3((XMVECTOR const&)diff);
	return sqrt(dotValue);
}

inline float XM_CALLCONV length(const Math::Vector4& vec1) noexcept {
	Math::Vector4 diff = vec1 * vec1;
	float dotValue = CombineVector4((XMVECTOR const&)diff);
	return sqrt(dotValue);
}

inline float XM_CALLCONV distance(const Math::Vector4& vec1, const Math::Vector4& vec2) noexcept {
	Math::Vector4 diff = vec1 - vec2;
	diff *= diff;
	float dotValue = CombineVector4((XMVECTOR const&)diff);
	return sqrt(dotValue);
}

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

VENGINE_DLL_COMMON Math::Matrix4 XM_CALLCONV mul(
	const Math::Matrix4& m1,
	const Math::Matrix4& m2) noexcept;

template<typename T>
constexpr inline T XM_CALLCONV Max(const T& a, const T& b) noexcept {
	return (((a) > (b)) ? (a) : (b));
}
template<typename T>
constexpr inline T XM_CALLCONV Max(std::initializer_list<T> const& vars) noexcept {
	if (vars.size() == 0) return T();
	T const* maxValue = vars.begin();
	for (T const* ptr = maxValue + 1; ptr != vars.end(); ++ptr) {
		if (*maxValue < *ptr)
			maxValue = ptr;
	}
	return *maxValue;
}
template<typename T>
constexpr inline T XM_CALLCONV Min(std::initializer_list<T> const& vars) noexcept {
	if (vars.size() == 0) return T();
	T const* minValue = vars.begin();
	for (T const* ptr = minValue + 1; ptr != vars.end(); ++ptr) {
		if (*minValue > *ptr)
			minValue = ptr;
	}
	return *minValue;
}
template<typename T>
constexpr inline T XM_CALLCONV Min(const T& a, const T& b) noexcept {
	return (((a) < (b)) ? (a) : (b));
}
template<>
inline Math::Vector3 XM_CALLCONV Max<Math::Vector3>(const Math::Vector3& vec1, const Math::Vector3& vec2) noexcept {
#ifdef _XM_NO_INTRINSICS_
	return Math::Vector3(XMVectorMax((FXMVECTOR&)vec1, (FXMVECTOR&)vec2));
#else
	return _mm_max_ps((XMVECTOR const&)vec1, (XMVECTOR const&)vec2);
#endif
}
template<>
inline Math::Vector4 XM_CALLCONV Max<Math::Vector4>(const Math::Vector4& vec1, const Math::Vector4& vec2) noexcept {
#ifdef _XM_NO_INTRINSICS_
	return Math::Vector4(XMVectorMax((FXMVECTOR&)vec1, (FXMVECTOR&)vec2));
#else
	return _mm_max_ps((XMVECTOR const&)vec1, (XMVECTOR const&)vec2);
#endif
}
template<>
inline Math::Vector3 XM_CALLCONV Min<Math::Vector3>(const Math::Vector3& vec1, const Math::Vector3& vec2) noexcept {
#ifdef _XM_NO_INTRINSICS_
	return Math::Vector3(XMVectorMin((FXMVECTOR&)vec1, (FXMVECTOR&)vec2));
#else
	return _mm_min_ps((XMVECTOR const&)vec1, (XMVECTOR const&)vec2);
#endif
}
template<>
inline Math::Vector4 XM_CALLCONV Min<Math::Vector4>(const Math::Vector4& vec1, const Math::Vector4& vec2) noexcept {
#ifdef _XM_NO_INTRINSICS_
	return Math::Vector4(XMVectorMin((FXMVECTOR&)vec1, (FXMVECTOR&)vec2));
#else
	return _mm_min_ps((XMVECTOR const&)vec1, (XMVECTOR const&)vec2);
#endif
}

inline Math::Vector4 XM_CALLCONV floor(const Math::Vector4& c) noexcept {
	XMVECTOR const& V = (XMVECTOR const&)c;
#if defined(_XM_NO_INTRINSICS_)
	XMVECTORF32 Result = {{{floorf(V.vector4_f32[0]),
							floorf(V.vector4_f32[1]),
							floorf(V.vector4_f32[2]),
							floorf(V.vector4_f32[3])}}};
	return Result.v;
#elif defined(_XM_ARM_NEON_INTRINSICS_)
#if defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64)
	return vrndmq_f32(V);
#else
	float32x4_t vTest = vabsq_f32(V);
	vTest = vcltq_f32(vTest, g_XMNoFraction);
	// Truncate
	int32x4_t vInt = vcvtq_s32_f32(V);
	XMVECTOR vResult = vcvtq_f32_s32(vInt);
	XMVECTOR vLarger = vcgtq_f32(vResult, V);
	// 0 -> 0, 0xffffffff -> -1.0f
	vLarger = vcvtq_f32_s32(vLarger);
	vResult = vaddq_f32(vResult, vLarger);
	// All numbers less than 8388608 will use the round to int32_t
	// All others, use the ORIGINAL value
	return vbslq_f32(vTest, vResult, V);
#endif
#elif defined(_XM_SSE4_INTRINSICS_)
	return _mm_floor_ps(V);
#elif defined(_XM_SSE_INTRINSICS_)
	// To handle NAN, INF and numbers greater than 8388608, use masking
	__m128i vTest = _mm_and_si128(_mm_castps_si128(V), g_XMAbsMask);
	vTest = _mm_cmplt_epi32(vTest, g_XMNoFraction);
	// Truncate
	__m128i vInt = _mm_cvttps_epi32(V);
	XMVECTOR vResult = _mm_cvtepi32_ps(vInt);
	__m128 vLarger = _mm_cmpgt_ps(vResult, V);
	// 0 -> 0, 0xffffffff -> -1.0f
	vLarger = _mm_cvtepi32_ps(_mm_castps_si128(vLarger));
	vResult = _mm_add_ps(vResult, vLarger);
	// All numbers less than 8388608 will use the round to int32_t
	vResult = _mm_and_ps(vResult, _mm_castsi128_ps(vTest));
	// All others, use the ORIGINAL value
	vTest = _mm_andnot_si128(vTest, _mm_castps_si128(V));
	vResult = _mm_or_ps(vResult, _mm_castsi128_ps(vTest));
	return vResult;
#endif
}
inline Math::Vector3 XM_CALLCONV floor(const Math::Vector3& c) noexcept {
	XMVECTOR const& V = (XMVECTOR const&)c;
#if defined(_XM_NO_INTRINSICS_)
	XMVECTORF32 Result = {{{floorf(V.vector4_f32[0]),
							floorf(V.vector4_f32[1]),
							floorf(V.vector4_f32[2]),
							floorf(V.vector4_f32[3])}}};
	return Result.v;
#elif defined(_XM_ARM_NEON_INTRINSICS_)
#if defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64)
	return vrndmq_f32(V);
#else
	float32x4_t vTest = vabsq_f32(V);
	vTest = vcltq_f32(vTest, g_XMNoFraction);
	// Truncate
	int32x4_t vInt = vcvtq_s32_f32(V);
	XMVECTOR vResult = vcvtq_f32_s32(vInt);
	XMVECTOR vLarger = vcgtq_f32(vResult, V);
	// 0 -> 0, 0xffffffff -> -1.0f
	vLarger = vcvtq_f32_s32(vLarger);
	vResult = vaddq_f32(vResult, vLarger);
	// All numbers less than 8388608 will use the round to int32_t
	// All others, use the ORIGINAL value
	return vbslq_f32(vTest, vResult, V);
#endif
#elif defined(_XM_SSE4_INTRINSICS_)
	return _mm_floor_ps(V);
#elif defined(_XM_SSE_INTRINSICS_)
	// To handle NAN, INF and numbers greater than 8388608, use masking
	__m128i vTest = _mm_and_si128(_mm_castps_si128(V), g_XMAbsMask);
	vTest = _mm_cmplt_epi32(vTest, g_XMNoFraction);
	// Truncate
	__m128i vInt = _mm_cvttps_epi32(V);
	XMVECTOR vResult = _mm_cvtepi32_ps(vInt);
	__m128 vLarger = _mm_cmpgt_ps(vResult, V);
	// 0 -> 0, 0xffffffff -> -1.0f
	vLarger = _mm_cvtepi32_ps(_mm_castps_si128(vLarger));
	vResult = _mm_add_ps(vResult, vLarger);
	// All numbers less than 8388608 will use the round to int32_t
	vResult = _mm_and_ps(vResult, _mm_castsi128_ps(vTest));
	// All others, use the ORIGINAL value
	vTest = _mm_andnot_si128(vTest, _mm_castps_si128(V));
	vResult = _mm_or_ps(vResult, _mm_castsi128_ps(vTest));
	return vResult;
#endif
}
inline Math::Vector4 XM_CALLCONV ceil(
	const Math::Vector4& c) noexcept {
	XMVECTOR const& V = (XMVECTOR const&)c;
#if defined(_XM_NO_INTRINSICS_)
	XMVECTORF32 Result = {{{ceilf(V.vector4_f32[0]),
							ceilf(V.vector4_f32[1]),
							ceilf(V.vector4_f32[2]),
							ceilf(V.vector4_f32[3])}}};
	return Result.v;
#elif defined(_XM_ARM_NEON_INTRINSICS_)
#if defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64)
	return vrndpq_f32(V);
#else
	float32x4_t vTest = vabsq_f32(V);
	vTest = vcltq_f32(vTest, g_XMNoFraction);
	// Truncate
	int32x4_t vInt = vcvtq_s32_f32(V);
	XMVECTOR vResult = vcvtq_f32_s32(vInt);
	XMVECTOR vSmaller = vcltq_f32(vResult, V);
	// 0 -> 0, 0xffffffff -> -1.0f
	vSmaller = vcvtq_f32_s32(vSmaller);
	vResult = vsubq_f32(vResult, vSmaller);
	// All numbers less than 8388608 will use the round to int32_t
	// All others, use the ORIGINAL value
	return vbslq_f32(vTest, vResult, V);
#endif
#elif defined(_XM_SSE4_INTRINSICS_)
	return _mm_ceil_ps(V);
#elif defined(_XM_SSE_INTRINSICS_)
	// To handle NAN, INF and numbers greater than 8388608, use masking
	__m128i vTest = _mm_and_si128(_mm_castps_si128(V), g_XMAbsMask);
	vTest = _mm_cmplt_epi32(vTest, g_XMNoFraction);
	// Truncate
	__m128i vInt = _mm_cvttps_epi32(V);
	XMVECTOR vResult = _mm_cvtepi32_ps(vInt);
	__m128 vSmaller = _mm_cmplt_ps(vResult, V);
	// 0 -> 0, 0xffffffff -> -1.0f
	vSmaller = _mm_cvtepi32_ps(_mm_castps_si128(vSmaller));
	vResult = _mm_sub_ps(vResult, vSmaller);
	// All numbers less than 8388608 will use the round to int32_t
	vResult = _mm_and_ps(vResult, _mm_castsi128_ps(vTest));
	// All others, use the ORIGINAL value
	vTest = _mm_andnot_si128(vTest, _mm_castps_si128(V));
	vResult = _mm_or_ps(vResult, _mm_castsi128_ps(vTest));
	return vResult;
#endif
}
inline Math::Vector3 XM_CALLCONV ceil(
	const Math::Vector3& c) noexcept {
	XMVECTOR const& V = (XMVECTOR const&)c;
#if defined(_XM_NO_INTRINSICS_)
	XMVECTORF32 Result = {{{ceilf(V.vector4_f32[0]),
							ceilf(V.vector4_f32[1]),
							ceilf(V.vector4_f32[2]),
							ceilf(V.vector4_f32[3])}}};
	return Result.v;
#elif defined(_XM_ARM_NEON_INTRINSICS_)
#if defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64)
	return vrndpq_f32(V);
#else
	float32x4_t vTest = vabsq_f32(V);
	vTest = vcltq_f32(vTest, g_XMNoFraction);
	// Truncate
	int32x4_t vInt = vcvtq_s32_f32(V);
	XMVECTOR vResult = vcvtq_f32_s32(vInt);
	XMVECTOR vSmaller = vcltq_f32(vResult, V);
	// 0 -> 0, 0xffffffff -> -1.0f
	vSmaller = vcvtq_f32_s32(vSmaller);
	vResult = vsubq_f32(vResult, vSmaller);
	// All numbers less than 8388608 will use the round to int32_t
	// All others, use the ORIGINAL value
	return vbslq_f32(vTest, vResult, V);
#endif
#elif defined(_XM_SSE4_INTRINSICS_)
	return _mm_ceil_ps(V);
#elif defined(_XM_SSE_INTRINSICS_)
	// To handle NAN, INF and numbers greater than 8388608, use masking
	__m128i vTest = _mm_and_si128(_mm_castps_si128(V), g_XMAbsMask);
	vTest = _mm_cmplt_epi32(vTest, g_XMNoFraction);
	// Truncate
	__m128i vInt = _mm_cvttps_epi32(V);
	XMVECTOR vResult = _mm_cvtepi32_ps(vInt);
	__m128 vSmaller = _mm_cmplt_ps(vResult, V);
	// 0 -> 0, 0xffffffff -> -1.0f
	vSmaller = _mm_cvtepi32_ps(_mm_castps_si128(vSmaller));
	vResult = _mm_sub_ps(vResult, vSmaller);
	// All numbers less than 8388608 will use the round to int32_t
	vResult = _mm_and_ps(vResult, _mm_castsi128_ps(vTest));
	// All others, use the ORIGINAL value
	vTest = _mm_andnot_si128(vTest, _mm_castps_si128(V));
	vResult = _mm_or_ps(vResult, _mm_castsi128_ps(vTest));
	return vResult;
#endif
}

VENGINE_DLL_COMMON Math::Matrix4 XM_CALLCONV transpose(const Math::Matrix4& m) noexcept;

inline Math::Vector3 XM_CALLCONV pow(const Math::Vector3& v1, const Math::Vector3& v2) noexcept {
	XMVECTOR const& V1 = (XMVECTOR const&)v1;
	XMVECTOR const& V2 = (XMVECTOR const&)v2;
#if defined(_XM_NO_INTRINSICS_)

	XMVECTORF32 Result = {{{powf(V1.vector4_f32[0], V2.vector4_f32[0]),
							powf(V1.vector4_f32[1], V2.vector4_f32[1]),
							powf(V1.vector4_f32[2], V2.vector4_f32[2]),
							powf(V1.vector4_f32[3], V2.vector4_f32[3])}}};
	return Result.v;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMVECTORF32 vResult = {{{powf(vgetq_lane_f32(V1, 0), vgetq_lane_f32(V2, 0)),
							 powf(vgetq_lane_f32(V1, 1), vgetq_lane_f32(V2, 1)),
							 powf(vgetq_lane_f32(V1, 2), vgetq_lane_f32(V2, 2)),
							 powf(vgetq_lane_f32(V1, 3), vgetq_lane_f32(V2, 3))}}};
	return vResult.v;
#elif defined(_XM_SSE_INTRINSICS_)
	__declspec(align(16)) float a[4];
	__declspec(align(16)) float b[4];
	_mm_store_ps(a, V1);
	_mm_store_ps(b, V2);
	XMVECTOR vResult = _mm_setr_ps(
		powf(a[0], b[0]),
		powf(a[1], b[1]),
		powf(a[2], b[2]),
		powf(a[3], b[3]));
	return vResult;
#endif
}

inline Math::Vector3 XM_CALLCONV pow(const Math::Vector3& v1, float v2) noexcept {
	XMVECTOR const& V1 = (XMVECTOR const&)v1;
	XMVECTOR V2 = XMVectorReplicate(v2);
	//XMVECTOR const& V2 = (XMVECTOR const&)v2;
#if defined(_XM_NO_INTRINSICS_)

	XMVECTORF32 Result = {{{powf(V1.vector4_f32[0], V2.vector4_f32[0]),
							powf(V1.vector4_f32[1], V2.vector4_f32[1]),
							powf(V1.vector4_f32[2], V2.vector4_f32[2]),
							powf(V1.vector4_f32[3], V2.vector4_f32[3])}}};
	return Result.v;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMVECTORF32 vResult = {{{powf(vgetq_lane_f32(V1, 0), vgetq_lane_f32(V2, 0)),
							 powf(vgetq_lane_f32(V1, 1), vgetq_lane_f32(V2, 1)),
							 powf(vgetq_lane_f32(V1, 2), vgetq_lane_f32(V2, 2)),
							 powf(vgetq_lane_f32(V1, 3), vgetq_lane_f32(V2, 3))}}};
	return vResult.v;
#elif defined(_XM_SSE_INTRINSICS_)
	__declspec(align(16)) float a[4];
	__declspec(align(16)) float b[4];
	_mm_store_ps(a, V1);
	_mm_store_ps(b, V2);
	XMVECTOR vResult = _mm_setr_ps(
		powf(a[0], b[0]),
		powf(a[1], b[1]),
		powf(a[2], b[2]),
		powf(a[3], b[3]));
	return vResult;
#endif
}

inline Math::Vector4 XM_CALLCONV pow(const Math::Vector4& v1, const Math::Vector4& v2) noexcept {
	XMVECTOR const& V1 = (XMVECTOR const&)v1;
	XMVECTOR const& V2 = (XMVECTOR const&)v2;
#if defined(_XM_NO_INTRINSICS_)

	XMVECTORF32 Result = {{{powf(V1.vector4_f32[0], V2.vector4_f32[0]),
							powf(V1.vector4_f32[1], V2.vector4_f32[1]),
							powf(V1.vector4_f32[2], V2.vector4_f32[2]),
							powf(V1.vector4_f32[3], V2.vector4_f32[3])}}};
	return Result.v;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMVECTORF32 vResult = {{{powf(vgetq_lane_f32(V1, 0), vgetq_lane_f32(V2, 0)),
							 powf(vgetq_lane_f32(V1, 1), vgetq_lane_f32(V2, 1)),
							 powf(vgetq_lane_f32(V1, 2), vgetq_lane_f32(V2, 2)),
							 powf(vgetq_lane_f32(V1, 3), vgetq_lane_f32(V2, 3))}}};
	return vResult.v;
#elif defined(_XM_SSE_INTRINSICS_)
	__declspec(align(16)) float a[4];
	__declspec(align(16)) float b[4];
	_mm_store_ps(a, V1);
	_mm_store_ps(b, V2);
	XMVECTOR vResult = _mm_setr_ps(
		powf(a[0], b[0]),
		powf(a[1], b[1]),
		powf(a[2], b[2]),
		powf(a[3], b[3]));
	return vResult;
#endif
}

inline Math::Vector4 XM_CALLCONV pow(const Math::Vector4& v1, float v2) {
	XMVECTOR const& V1 = (XMVECTOR const&)v1;
	XMVECTOR V2 = XMVectorReplicate(v2);
#if defined(_XM_NO_INTRINSICS_)

	XMVECTORF32 Result = {{{powf(V1.vector4_f32[0], V2.vector4_f32[0]),
							powf(V1.vector4_f32[1], V2.vector4_f32[1]),
							powf(V1.vector4_f32[2], V2.vector4_f32[2]),
							powf(V1.vector4_f32[3], V2.vector4_f32[3])}}};
	return Result.v;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMVECTORF32 vResult = {{{powf(vgetq_lane_f32(V1, 0), vgetq_lane_f32(V2, 0)),
							 powf(vgetq_lane_f32(V1, 1), vgetq_lane_f32(V2, 1)),
							 powf(vgetq_lane_f32(V1, 2), vgetq_lane_f32(V2, 2)),
							 powf(vgetq_lane_f32(V1, 3), vgetq_lane_f32(V2, 3))}}};
	return vResult.v;
#elif defined(_XM_SSE_INTRINSICS_)
	__declspec(align(16)) float a[4];
	__declspec(align(16)) float b[4];
	_mm_store_ps(a, V1);
	_mm_store_ps(b, V2);
	XMVECTOR vResult = _mm_setr_ps(
		powf(a[0], b[0]),
		powf(a[1], b[1]),
		powf(a[2], b[2]),
		powf(a[3], b[3]));
	return vResult;
#endif
}
VENGINE_DLL_COMMON Math::Matrix3 XM_CALLCONV transpose(const Math::Matrix3& m) noexcept;

VENGINE_DLL_COMMON Math::Matrix4 XM_CALLCONV inverse(const Math::Matrix4& m) noexcept;

VENGINE_DLL_COMMON Math::Matrix4 XM_CALLCONV QuaternionToMatrix(const Math::Vector4& q) noexcept;

VENGINE_DLL_COMMON Math::Matrix4 XM_CALLCONV GetTransformMatrix(const Math::Vector3& right, const Math::Vector3& up, const Math::Vector3& forward, const Math::Vector3& position) noexcept;
VENGINE_DLL_COMMON Math::Matrix4 XM_CALLCONV GetTransposedTransformMatrix(const Math::Vector3& right, const Math::Vector3& up, const Math::Vector3& forward, const Math::Vector3& position) noexcept;
VENGINE_DLL_COMMON Math::Vector4 XM_CALLCONV cross(const Math::Vector4& v1, const Math::Vector4& v2, const Math::Vector4& v3) noexcept;
VENGINE_DLL_COMMON Math::Vector3 XM_CALLCONV cross(const Math::Vector3& v1, const Math::Vector3& v2) noexcept;
VENGINE_DLL_COMMON Math::Vector4 XM_CALLCONV normalize(const Math::Vector4& v) noexcept;
VENGINE_DLL_COMMON Math::Vector3 XM_CALLCONV normalize(const Math::Vector3& v) noexcept;
VENGINE_DLL_COMMON Math::Matrix4 XM_CALLCONV GetInverseTransformMatrix(const Math::Vector3& right, const Math::Vector3& up, const Math::Vector3& forward, const Math::Vector3& position) noexcept;
struct double2 {
	double x;
	double y;
	double2(double x, double y) : x(x), y(y) {}
	double2() : x(0), y(0) {}
	double2(double x) : x(x), y(x) {}
};

struct double3 {
	double x, y, z;
	double3(double x, double y, double z) : x(x), y(y), z(z) {}
	double3() : x(0), y(0), z(0) {
	}
	double3(double x) : x(x), y(x), z(x) {}
};

struct double4 {
	double x, y, z, w;
	double4(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {}
	double4(double x) : x(x), y(x), z(x), w(x) {}
	double4() : x(0), y(0), z(0), w(0) {
	}
};

inline float2 XM_CALLCONV operator+(const float2& a, const float2& b) {
	return float2(a.x + b.x, a.y + b.y);
}
inline float3 XM_CALLCONV operator+(const float3& a, const float3& b) {
	return float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline float4 XM_CALLCONV operator+(const float4& a, const float4& b) {
	return float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline double2 XM_CALLCONV operator+(const double2& a, const double2& b) {
	return double2(a.x + b.x, a.y + b.y);
}
inline double3 XM_CALLCONV operator+(const double3& a, const double3& b) {
	return double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline double4 XM_CALLCONV operator+(const double4& a, const double4& b) {
	return double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline int2 XM_CALLCONV operator+(const int2& a, const int2& b) {
	return int2(a.x + b.x, a.y + b.y);
}
inline int3 XM_CALLCONV operator+(const int3& a, const int3& b) {
	return int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline int4 XM_CALLCONV operator+(const int4& a, const int4& b) {
	return int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline uint2 XM_CALLCONV operator+(const uint2& a, const uint2& b) {
	return uint2(a.x + b.x, a.y + b.y);
}
inline uint3 XM_CALLCONV operator+(const uint3& a, const uint3& b) {
	return uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline uint4 XM_CALLCONV operator+(const uint4& a, const uint4& b) {
	return uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline float2 XM_CALLCONV operator-(const float2& a, const float2& b) {
	return float2(a.x - b.x, a.y - b.y);
}
inline float3 XM_CALLCONV operator-(const float3& a, const float3& b) {
	return float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline float4 XM_CALLCONV operator-(const float4& a, const float4& b) {
	return float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline double2 XM_CALLCONV operator-(const double2& a, const double2& b) {
	return double2(a.x - b.x, a.y - b.y);
}
inline double3 XM_CALLCONV operator-(const double3& a, const double3& b) {
	return double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline double4 XM_CALLCONV operator-(const double4& a, const double4& b) {
	return double4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline int2 XM_CALLCONV operator-(const int2& a, const int2& b) {
	return int2(a.x - b.x, a.y - b.y);
}
inline int3 XM_CALLCONV operator-(const int3& a, const int3& b) {
	return int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline int4 XM_CALLCONV operator-(const int4& a, const int4& b) {
	return int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline uint2 XM_CALLCONV operator-(const uint2& a, const uint2& b) {
	return uint2(a.x - b.x, a.y - b.y);
}
inline uint3 XM_CALLCONV operator-(const uint3& a, const uint3& b) {
	return uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline uint4 XM_CALLCONV operator-(const uint4& a, const uint4& b) {
	return uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline float2 XM_CALLCONV operator*(const float2& a, const float2& b) {
	return float2(a.x * b.x, a.y * b.y);
}
inline float3 XM_CALLCONV operator*(const float3& a, const float3& b) {
	return float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline float4 XM_CALLCONV operator*(const float4& a, const float4& b) {
	return float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline double2 XM_CALLCONV operator*(const double2& a, const double2& b) {
	return double2(a.x * b.x, a.y * b.y);
}
inline double3 XM_CALLCONV operator*(const double3& a, const double3& b) {
	return double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline double4 XM_CALLCONV operator*(const double4& a, const double4& b) {
	return double4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline int2 XM_CALLCONV operator*(const int2& a, const int2& b) {
	return int2(a.x * b.x, a.y * b.y);
}
inline int3 XM_CALLCONV operator*(const int3& a, const int3& b) {
	return int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline int4 XM_CALLCONV operator*(const int4& a, const int4& b) {
	return int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline uint2 XM_CALLCONV operator*(const uint2& a, const uint2& b) {
	return uint2(a.x * b.x, a.y * b.y);
}
inline uint3 XM_CALLCONV operator*(const uint3& a, const uint3& b) {
	return uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline uint4 XM_CALLCONV operator*(const uint4& a, const uint4& b) {
	return uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline float2 XM_CALLCONV operator/(const float2& a, const float2& b) {
	return float2(a.x / b.x, a.y / b.y);
}
inline float3 XM_CALLCONV operator/(const float3& a, const float3& b) {
	return float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline float4 XM_CALLCONV operator/(const float4& a, const float4& b) {
	return float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline double2 XM_CALLCONV operator/(const double2& a, const double2& b) {
	return double2(a.x / b.x, a.y / b.y);
}
inline double3 XM_CALLCONV operator/(const double3& a, const double3& b) {
	return double3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline double4 XM_CALLCONV operator/(const double4& a, const double4& b) {
	return double4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline int2 XM_CALLCONV operator/(const int2& a, const int2& b) {
	return int2(a.x / b.x, a.y / b.y);
}
inline int3 XM_CALLCONV operator/(const int3& a, const int3& b) {
	return int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline int4 XM_CALLCONV operator/(const int4& a, const int4& b) {
	return int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline uint2 XM_CALLCONV operator/(const uint2& a, const uint2& b) {
	return uint2(a.x / b.x, a.y / b.y);
}
inline uint3 XM_CALLCONV operator/(const uint3& a, const uint3& b) {
	return uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline uint4 XM_CALLCONV operator/(const uint4& a, const uint4& b) {
	return uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline float2& XM_CALLCONV operator+=(float2& a, const float2& b) {
	a.x += b.x;
	a.y += b.y;
	return a;
}
inline float3& XM_CALLCONV operator+=(float3& a, const float3& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}
inline float4& XM_CALLCONV operator+=(float4& a, const float4& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

inline double2& XM_CALLCONV operator+=(double2& a, const double2& b) {
	a.x += b.x;
	a.y += b.y;
	return a;
}
inline double3& XM_CALLCONV operator+=(double3& a, const double3& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}
inline double4& XM_CALLCONV operator+=(double4& a, const double4& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}
inline uint2& XM_CALLCONV operator+=(uint2& a, const uint2& b) {
	a.x += b.x;
	a.y += b.y;
	return a;
}
inline uint3& XM_CALLCONV operator+=(uint3& a, const uint3& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}
inline uint4& XM_CALLCONV operator+=(uint4& a, const uint4& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

inline int2& XM_CALLCONV operator+=(int2& a, const int2& b) {
	a.x += b.x;
	a.y += b.y;
	return a;
}
inline int3& XM_CALLCONV operator+=(int3& a, const int3& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}
inline int4& XM_CALLCONV operator+=(int4& a, const int4& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

inline float2& XM_CALLCONV operator-=(float2& a, const float2& b) {
	a.x -= b.x;
	a.y -= b.y;
	return a;
}
inline float3& XM_CALLCONV operator-=(float3& a, const float3& b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}
inline float4& XM_CALLCONV operator-=(float4& a, const float4& b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

inline double2& XM_CALLCONV operator-=(double2& a, const double2& b) {
	a.x -= b.x;
	a.y -= b.y;
	return a;
}
inline double3& XM_CALLCONV operator-=(double3& a, const double3& b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}
inline double4& XM_CALLCONV operator-=(double4& a, const double4& b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}
inline uint2& XM_CALLCONV operator-=(uint2& a, const uint2& b) {
	a.x -= b.x;
	a.y -= b.y;
	return a;
}
inline uint3& XM_CALLCONV operator-=(uint3& a, const uint3& b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}
inline uint4& XM_CALLCONV operator-=(uint4& a, const uint4& b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

inline int2& XM_CALLCONV operator-=(int2& a, const int2& b) {
	a.x -= b.x;
	a.y -= b.y;
	return a;
}
inline int3& XM_CALLCONV operator-=(int3& a, const int3& b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}
inline int4& XM_CALLCONV operator-=(int4& a, const int4& b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

inline float2& XM_CALLCONV operator*=(float2& a, const float2& b) {
	a.x *= b.x;
	a.y *= b.y;
	return a;
}
inline float3& XM_CALLCONV operator*=(float3& a, const float3& b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}
inline float4& XM_CALLCONV operator*=(float4& a, const float4& b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

inline double2& XM_CALLCONV operator*=(double2& a, const double2& b) {
	a.x *= b.x;
	a.y *= b.y;
	return a;
}
inline double3& XM_CALLCONV operator*=(double3& a, const double3& b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}
inline double4& XM_CALLCONV operator*=(double4& a, const double4& b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}
inline uint2& XM_CALLCONV operator*=(uint2& a, const uint2& b) {
	a.x *= b.x;
	a.y *= b.y;
	return a;
}
inline uint3& XM_CALLCONV operator*=(uint3& a, const uint3& b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}
inline uint4& XM_CALLCONV operator*=(uint4& a, const uint4& b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

inline int2& XM_CALLCONV operator*=(int2& a, const int2& b) {
	a.x *= b.x;
	a.y *= b.y;
	return a;
}
inline int3& XM_CALLCONV operator*=(int3& a, const int3& b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}
inline int4& XM_CALLCONV operator*=(int4& a, const int4& b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}
inline float2& XM_CALLCONV operator/=(float2& a, const float2& b) {
	a.x /= b.x;
	a.y /= b.y;
	return a;
}
inline float3& XM_CALLCONV operator/=(float3& a, const float3& b) {
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}
inline float4& XM_CALLCONV operator/=(float4& a, const float4& b) {
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

inline double2& XM_CALLCONV operator/=(double2& a, const double2& b) {
	a.x /= b.x;
	a.y /= b.y;
	return a;
}
inline double3& XM_CALLCONV operator/=(double3& a, const double3& b) {
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}
inline double4& XM_CALLCONV operator/=(double4& a, const double4& b) {
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}
inline uint2& XM_CALLCONV operator/=(uint2& a, const uint2& b) {
	a.x /= b.x;
	a.y /= b.y;
	return a;
}
inline uint3& XM_CALLCONV operator/=(uint3& a, const uint3& b) {
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}
inline uint4& XM_CALLCONV operator/=(uint4& a, const uint4& b) {
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

inline int2& XM_CALLCONV operator/=(int2& a, const int2& b) {
	a.x /= b.x;
	a.y /= b.y;
	return a;
}
inline int3& XM_CALLCONV operator/=(int3& a, const int3& b) {
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}
inline int4& XM_CALLCONV operator/=(int4& a, const int4& b) {
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}
namespace vstd {
template<>
struct hash<uint2> {
	inline uint64_t operator()(uint2 const& value) const noexcept {
		uint32_t const* ptr = (uint32_t const*)&value;
		return Hash::Int32ArrayHash(ptr, ptr + 2);
	}
};
template<>
struct hash<uint3> {
	inline uint64_t operator()(uint3 const& value) const noexcept {
		uint32_t const* ptr = (uint32_t const*)&value;
		return Hash::Int32ArrayHash(ptr, ptr + 3);
	}
};
template<>
struct hash<uint4> {
	inline uint64_t operator()(uint4 const& value) const noexcept {
		uint32_t const* ptr = (uint32_t const*)&value;
		return Hash::Int32ArrayHash(ptr, ptr + 4);
	}
};
template<>
struct hash<int2> {
	inline uint64_t operator()(int2 const& value) const noexcept {
		uint32_t const* ptr = (uint32_t const*)&value;
		return Hash::Int32ArrayHash(ptr, ptr + 2);
	}
};
template<>
struct hash<int3> {
	inline uint64_t operator()(int3 const& value) const noexcept {
		uint32_t const* ptr = (uint32_t const*)&value;
		return Hash::Int32ArrayHash(ptr, ptr + 3);
	}
};
template<>
struct hash<int4> {
	inline uint64_t operator()(int4 const& value) const noexcept {
		uint32_t const* ptr = (uint32_t const*)&value;
		return Hash::Int32ArrayHash(ptr, ptr + 4);
	}
};
}// namespace vstd
namespace std {
template<>
struct equal_to<uint2> {
	inline bool operator()(uint2 const& a, uint2 const& b) const noexcept {
		return a.x == b.x && a.y == b.y;
	}
};
template<>
struct equal_to<uint3> {
	inline bool operator()(uint3 const& a, uint3 const& b) const noexcept {
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}
};
template<>
struct equal_to<uint4> {
	inline bool operator()(uint4 const& a, uint4 const& b) const noexcept {
		return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
	}
};
template<>
struct equal_to<int2> {
	inline bool operator()(int2 const& a, int2 const& b) const noexcept {
		return a.x == b.x && a.y == b.y;
	}
};
template<>
struct equal_to<int3> {
	inline bool operator()(int3 const& a, int3 const& b) const noexcept {
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}
};
template<>
struct equal_to<int4> {
	inline bool operator()(int4 const& a, int4 const& b) const noexcept {
		return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
	}
};
}// namespace std
