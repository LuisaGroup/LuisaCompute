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

#include <Common/DXMath/Matrix3.h>
namespace Math
{
	__declspec(align(16)) class Matrix4
	{
	public:
		INLINE XM_CALLCONV  Matrix4() {}
		INLINE XM_CALLCONV  Matrix4(const Vector3& x, const Vector3& y, const Vector3& z, const Vector3& w)
		{
			m_mat.r[0] = SetWToZero(x); m_mat.r[1] = SetWToZero(y);
			m_mat.r[2] = SetWToZero(z); m_mat.r[3] = SetWToOne(w);
		}
		INLINE XM_CALLCONV  Matrix4(const Vector4& x, const Vector4& y, const Vector4& z, const Vector4& w) { m_mat.r[0] = x; m_mat.r[1] = y; m_mat.r[2] = z; m_mat.r[3] = w; }
		INLINE XM_CALLCONV  Matrix4(const Matrix4& mat) : m_mat(mat.m_mat) { }
		INLINE XM_CALLCONV  Matrix4(const Matrix3& mat)
		{
			m_mat.r[0] = SetWToZero(mat.GetX());
			m_mat.r[1] = SetWToZero(mat.GetY());
			m_mat.r[2] = SetWToZero(mat.GetZ());
			m_mat.r[3] = CreateWUnitVector();
		}
		INLINE XM_CALLCONV  Matrix4(const XMMATRIX& mat) : m_mat(mat) {}
		INLINE XM_CALLCONV  Matrix4(const Matrix3& xyz, const Vector3& w)
		{
			m_mat.r[0] = SetWToZero(xyz.GetX());
			m_mat.r[1] = SetWToZero(xyz.GetY());
			m_mat.r[2] = SetWToZero(xyz.GetZ());
			m_mat.r[3] = SetWToOne(w);
		}
		INLINE XM_CALLCONV  Matrix4(EIdentityTag) { m_mat = XMMatrixIdentity(); }
		INLINE XM_CALLCONV  Matrix4(EZeroTag) { m_mat.r[0] = m_mat.r[1] = m_mat.r[2] = m_mat.r[3] = SplatZero(); }
		INLINE XM_CALLCONV  Matrix4(const float4x4& f)
		{
			m_mat = XMLoadFloat4x4(&f);
		}

		INLINE  XM_CALLCONV  operator float4x4() const
		{
			float4x4 f;
			XMStoreFloat4x4(&f, m_mat);
			return f;
		}

		INLINE  const Matrix3& XM_CALLCONV  Get3x3() const { return (const Matrix3&)*this; }

		INLINE Vector4 XM_CALLCONV  GetX() const { return Vector4(m_mat.r[0]); }
		INLINE Vector4  XM_CALLCONV GetY() const { return Vector4(m_mat.r[1]); }
		INLINE Vector4 XM_CALLCONV  GetZ() const { return Vector4(m_mat.r[2]); }
		INLINE Vector4  XM_CALLCONV GetW() const { return Vector4(m_mat.r[3]); }

		INLINE void XM_CALLCONV  SetX(const Vector4& x) { m_mat.r[0] = x; }
		INLINE void XM_CALLCONV  SetY(const Vector4& y) { m_mat.r[1] = y; }
		INLINE void XM_CALLCONV  SetZ(const Vector4& z) { m_mat.r[2] = z; }
		INLINE void XM_CALLCONV  SetW(const Vector4& w) { m_mat.r[3] = w; }

		INLINE  XM_CALLCONV operator XMMATRIX& () const { return (XMMATRIX&)m_mat; }
		INLINE XMVECTOR& XM_CALLCONV operator[] (uint i)
		{
			return m_mat.r[i];
		}
		INLINE const XMVECTOR& XM_CALLCONV  operator[] (uint i) const
		{
			return m_mat.r[i];
		}

		INLINE Vector4 XM_CALLCONV  operator* (const Vector3& vec) const { return Vector4(XMVector3Transform(vec, m_mat)); }
		INLINE Vector4  XM_CALLCONV operator* (const Vector4& vec) const { return Vector4(XMVector4Transform(vec, m_mat)); }
		INLINE Matrix4  XM_CALLCONV operator* (const Matrix4& mat) const { return Matrix4(XMMatrixMultiply(mat, m_mat)); }

		static INLINE Matrix4 XM_CALLCONV  MakeScale(float scale) { return Matrix4(XMMatrixScaling(scale, scale, scale)); }
		static INLINE Matrix4 XM_CALLCONV  MakeScale(const Vector3& scale) { return Matrix4(XMMatrixScalingFromVector(scale)); }


	private:
		XMMATRIX m_mat;
	};
}