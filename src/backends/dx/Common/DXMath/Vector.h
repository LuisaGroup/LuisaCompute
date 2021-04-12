//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author:  James Stanard
//

#pragma once
#include <Common/DXMath/Scalar.h>
using namespace DirectX;
namespace Math {
struct bool4;
struct bool3 {
	bool x, y, z;
	inline bool3(bool x, bool y, bool z) noexcept : x(x), y(y), z(z) {
	}
	inline bool3(bool x) noexcept : x(x), y(x), z(x) {}
	inline bool3(bool4 b) noexcept;
	inline void operator=(bool4 b) noexcept;
	inline void operator=(bool3 b) noexcept {
		x = b.x;
		y = b.y;
		z = b.z;
	}
	inline bool3 operator||(bool3 b) noexcept {
		return bool3(b.x || x, b.y || y, b.z || z);
	}
	inline bool3 operator^(bool3 b) noexcept {
		return bool3(b.x ^ x, b.y ^ y, b.z ^ z);
	}
	inline bool3 operator&(bool3 b) noexcept {
		return bool3(b.x & x, b.y & y, b.z & z);
	}
	inline bool3 operator|(bool3 b) noexcept {
		return bool3(b.x | x, b.y | y, b.z | z);
	}
	inline bool3 operator&&(bool3 b) noexcept {
		return bool3(b.x && x, b.y && y, b.z && z);
	}
	inline bool3 operator==(bool3 b) noexcept {
		return bool3(b.x == x, b.y == y, b.z == z);
	}
	inline bool3 operator!=(bool3 b) noexcept {
		return bool3(b.x != x, b.y != y, b.z != z);
	}
	inline bool3 operator!() noexcept {
		return bool3(!x, !y, !z);
	}
};

struct bool4 {
	bool x, y, z, w;

	inline bool4(bool x, bool y, bool z, bool w) noexcept : x(x), y(y), z(z), w(w) {}
	inline bool4(bool x) noexcept : x(x), y(x), z(x), w(x) {}
	inline bool4(bool3 b) noexcept : x(b.x), y(b.y), z(b.z), w(false) {
	}
	inline void operator=(bool4 b) noexcept {
		x = b.x;
		y = b.y;
		z = b.z;
		w = b.w;
	}

	inline void operator=(bool3 b) noexcept {
		x = b.x;
		y = b.y;
		z = b.z;
		w = false;
	}
	inline bool4 operator||(bool4 b) noexcept {
		return bool4(b.x || x, b.y || y, b.z || z, b.w || w);
	}
	inline bool4 operator^(bool4 b) noexcept {
		return bool4(b.x ^ x, b.y ^ y, b.z ^ z, b.w ^ w);
	}
	inline bool4 operator&(bool4 b) noexcept {
		return bool4(b.x & x, b.y & y, b.z & z, b.w & w);
	}
	inline bool4 operator|(bool4 b) noexcept {
		return bool4(b.x | x, b.y | y, b.z | z, b.w | w);
	}
	inline bool4 operator&&(bool4 b) noexcept {
		return bool4(b.x && x, b.y && y, b.z && z, b.w && w);
	}
	inline bool4 operator==(bool4 b) noexcept {
		return bool4(b.x == x, b.y == y, b.z == z, b.w == w);
	}
	inline bool4 operator!=(bool4 b) noexcept {
		return bool4(b.x != x, b.y != y, b.z != z, b.w != w);
	}
	inline bool4 operator!() noexcept {
		return bool4(!x, !y, !z, !w);
	}
};

inline bool3::bool3(bool4 b) noexcept : x(b.x), y(b.y), z(b.z) {
}

inline void bool3::operator=(bool4 b) noexcept {
	x = b.x;
	y = b.y;
	z = b.z;
}

inline XMVECTOR XM_CALLCONV VectorGreater(
	const XMVECTOR& V1,
	const XMVECTOR& V2) noexcept {
#if defined(_XM_NO_INTRINSICS_)

	XMVECTORU32 Control = {{{(V1.vector4_f32[0] > V2.vector4_f32[0]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[1] > V2.vector4_f32[1]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[2] > V2.vector4_f32[2]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[3] > V2.vector4_f32[3]) ? 0xFFFFFFFF : 0}}};
	return Control.v;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	return vcgtq_f32(V1, V2);
#elif defined(_XM_SSE_INTRINSICS_)
	return _mm_cmpgt_ps(V1, V2);
#endif
}

inline XMVECTOR XM_CALLCONV VectorGreaterEqual(
	const XMVECTOR& V1,
	const XMVECTOR& V2) noexcept {
#if defined(_XM_NO_INTRINSICS_)

	XMVECTORU32 Control = {{{(V1.vector4_f32[0] >= V2.vector4_f32[0]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[1] >= V2.vector4_f32[1]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[2] >= V2.vector4_f32[2]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[3] >= V2.vector4_f32[3]) ? 0xFFFFFFFF : 0}}};
	return Control.v;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	return vcgeq_f32(V1, V2);
#elif defined(_XM_SSE_INTRINSICS_)
	return _mm_cmpge_ps(V1, V2);
#endif
}

inline XMVECTOR XM_CALLCONV VectorLess(
	const XMVECTOR& V1,
	const XMVECTOR& V2) noexcept {
#if defined(_XM_NO_INTRINSICS_)

	XMVECTORU32 Control = {{{(V1.vector4_f32[0] < V2.vector4_f32[0]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[1] < V2.vector4_f32[1]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[2] < V2.vector4_f32[2]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[3] < V2.vector4_f32[3]) ? 0xFFFFFFFF : 0}}};
	return Control.v;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	return vcltq_f32(V1, V2);
#elif defined(_XM_SSE_INTRINSICS_)
	return _mm_cmplt_ps(V1, V2);
#endif
}

inline XMVECTOR XM_CALLCONV VectorLessEqual(
	const XMVECTOR& V1,
	const XMVECTOR& V2) noexcept {
#if defined(_XM_NO_INTRINSICS_)

	XMVECTORU32 Control = {{{(V1.vector4_f32[0] <= V2.vector4_f32[0]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[1] <= V2.vector4_f32[1]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[2] <= V2.vector4_f32[2]) ? 0xFFFFFFFF : 0,
							 (V1.vector4_f32[3] <= V2.vector4_f32[3]) ? 0xFFFFFFFF : 0}}};
	return Control.v;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	return vcleq_f32(V1, V2);
#elif defined(_XM_SSE_INTRINSICS_)
	return _mm_cmple_ps(V1, V2);
#endif
}

inline XMVECTOR XM_CALLCONV VectorEqual(
	const XMVECTOR& V1,
	const XMVECTOR& V2) noexcept {
#if defined(_XM_NO_INTRINSICS_)

	XMVECTORU32 Control = {{{
		(V1.vector4_f32[0] == V2.vector4_f32[0]) ? 0xFFFFFFFF : 0,
		(V1.vector4_f32[1] == V2.vector4_f32[1]) ? 0xFFFFFFFF : 0,
		(V1.vector4_f32[2] == V2.vector4_f32[2]) ? 0xFFFFFFFF : 0,
		(V1.vector4_f32[3] == V2.vector4_f32[3]) ? 0xFFFFFFFF : 0,
	}}};
	return Control.v;

#elif defined(_XM_ARM_NEON_INTRINSICS_)
	return vceqq_f32(V1, V2);
#elif defined(_XM_SSE_INTRINSICS_)
	return _mm_cmpeq_ps(V1, V2);
#endif
}
INLINE bool3 GetBool3(const XMVECTOR& vec) noexcept {
	return bool3(
		vec.m128_u32[0] != 0,
		vec.m128_u32[1] != 0,
		vec.m128_u32[2] != 0);
}
INLINE bool4 GetBool4(const XMVECTOR& vec) noexcept {
	return bool4(
		vec.m128_u32[0] != 0,
		vec.m128_u32[1] != 0,
		vec.m128_u32[2] != 0,
		vec.m128_u32[3] != 0);
}
class Vector4;
class Matrix3;
// A 3-vector with an unspecified fourth component.  Depending on the context, the W can be 0 or 1, but both are implicit.
// The actual value of the fourth component is undefined for performance reasons.
class Vector3 {
	friend class Matrix3;
	friend class Vector4;

public:
	INLINE Vector3() noexcept {}
	INLINE Vector3(float x, float y, float z) noexcept : m_vec(XMVectorSet(x, y, z, 0)) {}
	INLINE Vector3(const float3& v) noexcept : m_vec(XMLoadFloat3(&v)) {}
	INLINE Vector3(const Vector3& v) noexcept : m_vec(v) {}
	INLINE Vector3(const Scalar& s) noexcept : m_vec(s) {
		m_vec.m128_f32[3] = 0;
	}
	INLINE Vector3(const Vector4& v) noexcept;
	INLINE void operator=(const Vector4& v) noexcept;
	INLINE Vector3(FXMVECTOR& vec) noexcept : m_vec(vec) {
		m_vec.m128_f32[3] = 0;
	}

	INLINE operator const XMVECTOR&() const noexcept { return m_vec; }

	INLINE float GetX() const noexcept { return m_vec.m128_f32[0]; }
	INLINE float GetY() const noexcept { return m_vec.m128_f32[1]; }
	INLINE float GetZ() const noexcept { return m_vec.m128_f32[2]; }
	INLINE void SetX(float x) noexcept { m_vec.m128_f32[0] = x; }
	INLINE void SetY(float y) noexcept { m_vec.m128_f32[1] = y; }
	INLINE void SetZ(float z) noexcept { m_vec.m128_f32[2] = z; }

	INLINE Vector3 operator-() const noexcept { return Vector3(XMVectorNegate(m_vec)); }
	INLINE Vector3 operator+(const Vector3& v2) const noexcept { return Vector3(XMVectorAdd(m_vec, v2)); }
	INLINE Vector3 operator-(const Vector3& v2) const noexcept { return Vector3(XMVectorSubtract(m_vec, v2)); }
	INLINE Vector3 operator+(const Scalar& v2) const noexcept { return Vector3(XMVectorAdd(m_vec, v2)); }
	INLINE Vector3 operator-(const Scalar& v2) const noexcept { return Vector3(XMVectorSubtract(m_vec, v2)); }
	INLINE Vector3 operator*(const Vector3& v2) const noexcept { return Vector3(XMVectorMultiply(m_vec, v2)); }
	INLINE Vector3 operator/(const Vector3& v2) const noexcept { return Vector3(XMVectorDivide(m_vec, v2)); }
	INLINE Vector3 operator*(const Scalar& v2) const noexcept { return *this * Vector3(v2); }
	INLINE Vector3 operator/(const Scalar& v2) const noexcept { return *this / Vector3(v2); }
	INLINE Vector3 operator*(float v2) const noexcept { return *this * Scalar(v2); }
	INLINE Vector3 operator/(float v2) const noexcept { return *this / Scalar(v2); }

	INLINE Vector3& operator+=(const Vector3& v) noexcept {
		*this = *this + v;
		return *this;
	}
	INLINE Vector3& operator+=(float v) noexcept {
		*this = *this + v;
		return *this;
	}
	INLINE Vector3& operator-=(const Vector3& v) noexcept {
		*this = *this - v;
		return *this;
	}
	INLINE Vector3& operator-=(float v) noexcept {
		*this = *this - v;
		return *this;
	}
	INLINE Vector3& operator*=(const Vector3& v) noexcept {
		*this = *this * v;
		return *this;
	}
	INLINE Vector3& operator*=(float v) noexcept {
		*this = *this * v;
		return *this;
	}
	INLINE Vector3& operator/=(const Vector3& v) noexcept {
		*this = *this / v;
		return *this;
	}
	INLINE Vector3& operator/=(float v) noexcept {
		*this = *this / v;
		return *this;
	}

	INLINE friend Vector3 operator*(const Scalar& v1, const Vector3& v2) noexcept { return Vector3(v1) * v2; }
	INLINE friend Vector3 operator/(const Scalar& v1, const Vector3& v2) noexcept { return Vector3(v1) / v2; }
	INLINE friend Vector3 operator*(float v1, const Vector3& v2) noexcept { return Scalar(v1) * v2; }
	INLINE friend Vector3 operator/(float v1, const Vector3& v2) noexcept { return Scalar(v1) / v2; }
	INLINE Vector3 operator<(const Vector3& a) const noexcept {
		return VectorLess(m_vec, a.m_vec);
	}
	INLINE Vector3 operator>(const Vector3& a) const noexcept {
		return VectorGreater(m_vec, a.m_vec);
	}
	INLINE Vector3 operator==(const Vector3& a) const noexcept {
		return VectorEqual(m_vec, a.m_vec);
	}

	INLINE Vector3 operator<=(const Vector3& a) const noexcept {
		return VectorLessEqual(m_vec, a.m_vec);
	}

	INLINE Vector3 operator>=(const Vector3& a) const noexcept {
		return VectorGreaterEqual(m_vec, a.m_vec);
	}
	INLINE operator float4() const noexcept {
		return {m_vec.m128_f32[0], m_vec.m128_f32[1], m_vec.m128_f32[2], 0};
	}
	INLINE operator float3() const noexcept {
		return {m_vec.m128_f32[0], m_vec.m128_f32[1], m_vec.m128_f32[2]};
	}
	INLINE float& operator[](uint index) noexcept {
		return m_vec.m128_f32[index];
	}
	INLINE float const& operator[](uint index) const noexcept {
		return m_vec.m128_f32[index];
	}

protected:
	XMVECTOR m_vec;
};

// A 4-vector, completely defined.
class Vector4 {
public:
	friend class Vector3;
	INLINE Vector4() noexcept {}
	INLINE Vector4(float x, float y, float z, float w) noexcept : m_vec(XMVectorSet(x, y, z, w)) {}
	INLINE Vector4(const Vector3& xyz, float w) noexcept : m_vec(XMVectorSetW(xyz, w)) {}
	INLINE Vector4(const Vector4& v) noexcept : m_vec(v) {}
	INLINE Vector4(const Scalar& s) noexcept : m_vec(s) {}
	INLINE Vector4(const Vector3& xyz) noexcept : m_vec(SetWToOne(xyz)) {}
	INLINE void operator=(const Vector3& xyz) {
		m_vec = SetWToOne(xyz);
	}
	INLINE Vector4(const FXMVECTOR& vec) noexcept : m_vec(vec) {}
	INLINE Vector4(const float4& flt) noexcept : m_vec(XMLoadFloat4(&flt)) {}

	INLINE operator const XMVECTOR&() const noexcept { return m_vec; }

	INLINE float GetX() const noexcept { return m_vec.m128_f32[0]; }
	INLINE float GetY() const noexcept { return m_vec.m128_f32[1]; }
	INLINE float GetZ() const noexcept { return m_vec.m128_f32[2]; }
	INLINE float GetW() const noexcept { return m_vec.m128_f32[3]; }
	INLINE void SetX(float x) noexcept { m_vec.m128_f32[0] = x; }
	INLINE void SetY(float y) noexcept { m_vec.m128_f32[1] = y; }
	INLINE void SetZ(float z) noexcept { m_vec.m128_f32[2] = z; }
	INLINE void SetW(float w) noexcept { m_vec.m128_f32[3] = w; }

	INLINE Vector4 operator-() const noexcept { return Vector4(XMVectorNegate(m_vec)); }
	INLINE Vector4 operator+(const Vector4& v2) const noexcept { return Vector4(XMVectorAdd(m_vec, v2)); }
	INLINE Vector4 operator-(const Vector4& v2) const noexcept { return Vector4(XMVectorSubtract(m_vec, v2)); }
	INLINE Vector4 operator*(const Vector4& v2) const noexcept { return Vector4(XMVectorMultiply(m_vec, v2)); }
	INLINE Vector4 operator/(const Vector4& v2) const noexcept { return Vector4(XMVectorDivide(m_vec, v2)); }
	INLINE Vector4 operator*(const Scalar& v2) const noexcept { return *this * Vector4(v2); }
	INLINE Vector4 operator/(const Scalar& v2) const noexcept { return *this / Vector4(v2); }
	INLINE Vector4 operator+(const Scalar& v2) const noexcept { return *this + Vector4(v2); }
	INLINE Vector4 operator-(const Scalar& v2) const noexcept { return *this - Vector4(v2); }
	INLINE Vector4 operator*(float v2) const noexcept { return *this * Scalar(v2); }
	INLINE Vector4 operator/(float v2) const noexcept { return *this / Scalar(v2); }
	INLINE Vector4 operator+(float v2) const noexcept { return *this + Scalar(v2); }
	INLINE Vector4 operator-(float v2) const noexcept { return *this - Scalar(v2); }

	INLINE Vector4& operator*=(float v2) noexcept {
		*this = *this * Scalar(v2);
		return *this;
	}
	INLINE Vector4& operator/=(float v2) noexcept {
		*this = *this / Scalar(v2);
		return *this;
	}
	INLINE Vector4& operator+=(float v2) noexcept {
		*this = *this + Scalar(v2);
		return *this;
	}
	INLINE Vector4& operator-=(float v2) noexcept {
		*this = *this - Scalar(v2);
		return *this;
	}
	INLINE Vector4& operator+=(const Vector4& v) noexcept {
		*this = *this + v;
		return *this;
	}
	INLINE Vector4& operator-=(const Vector4& v) noexcept {
		*this = *this - v;
		return *this;
	}
	INLINE Vector4& operator*=(const Vector4& v) noexcept {
		*this = *this * v;
		return *this;
	}
	INLINE Vector4& operator/=(const Vector4& v) noexcept {
		*this = *this / v;
		return *this;
	}

	INLINE friend Vector4 operator*(const Scalar& v1, const Vector4& v2) noexcept { return Vector4(v1) * v2; }
	INLINE friend Vector4 operator/(const Scalar& v1, const Vector4& v2) noexcept { return Vector4(v1) / v2; }
	INLINE friend Vector4 operator*(float v1, const Vector4& v2) noexcept { return Scalar(v1) * v2; }
	INLINE friend Vector4 operator/(float v1, const Vector4& v2) noexcept { return Scalar(v1) / v2; }

	INLINE operator float4() const noexcept {
		return {m_vec.m128_f32[0], m_vec.m128_f32[1], m_vec.m128_f32[2], m_vec.m128_f32[3]};
	}
	INLINE operator float3() const noexcept {
		return {m_vec.m128_f32[0], m_vec.m128_f32[1], m_vec.m128_f32[2]};
	}

	INLINE Vector4 operator<(const Vector4& a) const noexcept {
		return VectorLess(m_vec, a.m_vec);
	}
	INLINE Vector4 operator>(const Vector4& a) const noexcept {
		return VectorGreater(m_vec, a.m_vec);
	}
	INLINE Vector4 operator==(const Vector4& a) const noexcept {
		return VectorEqual(m_vec, a.m_vec);
	}

	INLINE Vector4 operator<=(const Vector4& a) const noexcept {
		return VectorLessEqual(m_vec, a.m_vec);
	}

	INLINE Vector4 operator>=(const Vector4& a) const noexcept {
		return VectorGreaterEqual(m_vec, a.m_vec);
	}
	INLINE operator Vector3() const noexcept {
		return Vector3(m_vec);
	}
	INLINE float& operator[](uint index) noexcept {
		return m_vec.m128_f32[index];
	}
	INLINE float const& operator[](uint index) const noexcept {
		return m_vec.m128_f32[index];
	}

protected:
	XMVECTOR m_vec;
};

INLINE Vector3::Vector3(const Vector4& v) noexcept {
	m_vec = SetWToZero(v);
}
INLINE void Vector3::operator=(const Vector4& v) noexcept {
	m_vec = SetWToZero(v);
}

}// namespace Math