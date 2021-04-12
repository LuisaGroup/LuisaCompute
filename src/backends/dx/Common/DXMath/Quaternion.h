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

#include <Common/DXMath/Vector.h>

namespace Math
{
	class Quaternion
	{
	public:
		INLINE XM_CALLCONV  Quaternion() { m_vec = XMQuaternionIdentity(); }
		INLINE XM_CALLCONV  Quaternion(const Vector3& axis, const Scalar& angle) { m_vec = XMQuaternionRotationAxis(axis, angle); }
		INLINE XM_CALLCONV  Quaternion(float pitch, float yaw, float roll) { m_vec = XMQuaternionRotationRollPitchYaw(pitch, yaw, roll); }
		INLINE explicit  XM_CALLCONV Quaternion(const XMMATRIX& matrix) { m_vec = XMQuaternionRotationMatrix(matrix); }
		INLINE explicit  XM_CALLCONV Quaternion(FXMVECTOR vec) { m_vec = vec; }
		INLINE explicit  XM_CALLCONV Quaternion(EIdentityTag) { m_vec = XMQuaternionIdentity(); }

		INLINE  XM_CALLCONV operator XMVECTOR() const { return m_vec; }

		INLINE Quaternion  XM_CALLCONV operator~ (void) const { return Quaternion(XMQuaternionConjugate(m_vec)); }
		INLINE Quaternion  XM_CALLCONV operator- (void) const { return Quaternion(XMVectorNegate(m_vec)); }

		INLINE Quaternion  XM_CALLCONV operator* (Quaternion rhs) const { return Quaternion(XMQuaternionMultiply(rhs, m_vec)); }
		INLINE Vector3  XM_CALLCONV operator* (Vector3 rhs) const { return Vector3(XMVector3Rotate(rhs, m_vec)); }

		INLINE Quaternion& XM_CALLCONV operator= (Quaternion rhs) { m_vec = rhs; return *this; }
		INLINE Quaternion& XM_CALLCONV  operator*= (Quaternion rhs) { *this = *this * rhs; return *this; }

	protected:
		XMVECTOR m_vec;
	};

}