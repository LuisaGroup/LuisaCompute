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

#include "Quaternion.h"

namespace Math
{
	// Represents a 3x3 matrix while occuping a 4x4 memory footprint.  The unused row and column are undefined but implicitly
	// (0, 0, 0, 1).  Constructing a Matrix4 will make those values explicit.
	__declspec(align(16)) class Matrix3
	{
	public:
		INLINE XM_CALLCONV  Matrix3() {}
		INLINE XM_CALLCONV  Matrix3(const Vector3& x, const Vector3& y, const Vector3& z) { m_mat[0] = x; m_mat[1] = y; m_mat[2] = z; }
		INLINE XM_CALLCONV  Matrix3(const Matrix3& m) { m_mat[0] = m.m_mat[0]; m_mat[1] = m.m_mat[1]; m_mat[2] = m.m_mat[2]; }
		INLINE XM_CALLCONV  Matrix3(Quaternion q) { *this = Matrix3(XMMatrixRotationQuaternion(q)); }
		INLINE explicit XM_CALLCONV  Matrix3(const XMMATRIX& m) { m_mat[0] = Vector3(m.r[0]); m_mat[1] = Vector3(m.r[1]); m_mat[2] = Vector3(m.r[2]); }
		INLINE explicit XM_CALLCONV  Matrix3(EIdentityTag) { m_mat[0] = Vector3(kXUnitVector); m_mat[1] = Vector3(kYUnitVector); m_mat[2] = Vector3(kZUnitVector); }
		INLINE explicit XM_CALLCONV  Matrix3(EZeroTag) { m_mat[0] = m_mat[1] = m_mat[2] = Vector3(kZero); }

		INLINE void XM_CALLCONV  SetX(const Vector3& x) { m_mat[0] = x; }
		INLINE void XM_CALLCONV  SetY(const Vector3& y) { m_mat[1] = y; }
		INLINE void XM_CALLCONV  SetZ(const Vector3& z) { m_mat[2] = z; }

		INLINE Vector3 XM_CALLCONV  GetX() const { return m_mat[0]; }
		INLINE Vector3 XM_CALLCONV  GetY() const { return m_mat[1]; }
		INLINE Vector3 XM_CALLCONV  GetZ() const { return m_mat[2]; }

		static INLINE Matrix3 XM_CALLCONV  MakeXRotation(float angle) { return Matrix3(XMMatrixRotationX(angle)); }
		static INLINE Matrix3 XM_CALLCONV  MakeYRotation(float angle) { return Matrix3(XMMatrixRotationY(angle)); }
		static INLINE Matrix3 XM_CALLCONV  MakeZRotation(float angle) { return Matrix3(XMMatrixRotationZ(angle)); }
		static INLINE Matrix3 XM_CALLCONV  MakeScale(float scale) { return Matrix3(XMMatrixScaling(scale, scale, scale)); }
		static INLINE Matrix3 XM_CALLCONV  MakeScale(float sx, float sy, float sz) { return Matrix3(XMMatrixScaling(sx, sy, sz)); }
		static INLINE Matrix3 XM_CALLCONV  MakeScale(const Vector3& scale) { return Matrix3(XMMatrixScalingFromVector(scale)); }

		INLINE XM_CALLCONV  operator XMMATRIX&() const { return (XMMATRIX&)m_mat[0]; }
		INLINE XMVECTOR& XM_CALLCONV  operator[] (uint i)
		{
			return m_mat[i].m_vec;
		}
		INLINE const XMVECTOR& XM_CALLCONV  operator[] (uint i) const
		{
			return m_mat[i].m_vec;
		}
		INLINE Vector3 XM_CALLCONV  operator* (Vector3 vec) const { return Vector3(XMVector3TransformNormal(vec, *this)); }
		INLINE Matrix3 XM_CALLCONV  operator* (const Matrix3& mat) const { return Matrix3(*this * mat.GetX(), *this * mat.GetY(), *this * mat.GetZ()); }

	private:
		Vector3 m_mat[3];
	};

} // namespace Math