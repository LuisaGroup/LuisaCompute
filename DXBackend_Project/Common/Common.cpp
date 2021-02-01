#define NOMINMAX
#include <windows.h>
#include <wrl.h>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#include <memory>
//#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cassert>
#include "DLL.h"
#include "MCollection.h"
#include <comdef.h>
#include "DXMath/DXMath.h"
Math::Matrix4 XM_CALLCONV QuaternionToMatrix(const Math::Vector4& q) noexcept {
	static const XMVECTORF32 Constant1110 = {{{1.0f, 1.0f, 1.0f, 0.0f}}};
	XMVECTOR& Quaternion = (XMVECTOR&)q;
	XMVECTOR Q0 = _mm_add_ps(Quaternion, Quaternion);
	XMVECTOR Q1 = _mm_mul_ps(Quaternion, Q0);
	XMVECTOR V0 = XM_PERMUTE_PS(Q1, _MM_SHUFFLE(3, 0, 0, 1));
	V0 = _mm_and_ps(V0, g_XMMask3);
	XMVECTOR V1 = XM_PERMUTE_PS(Q1, _MM_SHUFFLE(3, 1, 2, 2));
	V1 = _mm_and_ps(V1, g_XMMask3);
	XMVECTOR R0 = _mm_sub_ps(Constant1110, V0);
	R0 = _mm_sub_ps(R0, V1);
	V0 = XM_PERMUTE_PS(Quaternion, _MM_SHUFFLE(3, 1, 0, 0));
	V1 = XM_PERMUTE_PS(Q0, _MM_SHUFFLE(3, 2, 1, 2));
	V0 = _mm_mul_ps(V0, V1);
	V1 = XM_PERMUTE_PS(Quaternion, _MM_SHUFFLE(3, 3, 3, 3));
	XMVECTOR V2 = XM_PERMUTE_PS(Q0, _MM_SHUFFLE(3, 0, 2, 1));
	V1 = _mm_mul_ps(V1, V2);
	XMVECTOR R1 = _mm_add_ps(V0, V1);
	XMVECTOR R2 = _mm_sub_ps(V0, V1);
	V0 = _mm_shuffle_ps(R1, R2, _MM_SHUFFLE(1, 0, 2, 1));
	V0 = XM_PERMUTE_PS(V0, _MM_SHUFFLE(1, 3, 2, 0));
	V1 = _mm_shuffle_ps(R1, R2, _MM_SHUFFLE(2, 2, 0, 0));
	V1 = XM_PERMUTE_PS(V1, _MM_SHUFFLE(2, 0, 2, 0));
	Q1 = _mm_shuffle_ps(R0, V0, _MM_SHUFFLE(1, 0, 3, 0));
	Q1 = XM_PERMUTE_PS(Q1, _MM_SHUFFLE(1, 3, 2, 0));
	XMMATRIX M;
	M.r[0] = Q1;
	Q1 = _mm_shuffle_ps(R0, V0, _MM_SHUFFLE(3, 2, 3, 1));
	Q1 = XM_PERMUTE_PS(Q1, _MM_SHUFFLE(1, 3, 0, 2));
	M.r[1] = Q1;
	Q1 = _mm_shuffle_ps(V1, R0, _MM_SHUFFLE(3, 2, 1, 0));
	M.r[2] = Q1;
	M.r[3] = g_XMIdentityR3;
	return transpose(M);
}
Math::Matrix4 XM_CALLCONV GetTransformMatrix(const Math::Vector3& right, const Math::Vector3& up, const Math::Vector3& forward, const Math::Vector3& position) noexcept {
	Math::Matrix4 target;
	target[0] = right;
	target[1] = up;
	target[2] = forward;
	target[3] = {0, 0, 0, 1};
	target[0].m128_f32[3] = position.GetX();
	target[1].m128_f32[3] = position.GetY();
	target[2].m128_f32[3] = position.GetZ();
	return transpose(target);
}
Math::Matrix4 XM_CALLCONV GetTransposedTransformMatrix(const Math::Vector3& right, const Math::Vector3& up, const Math::Vector3& forward, const Math::Vector3& position) noexcept {
	Math::Matrix4 target;
	target[0] = right;
	target[1] = up;
	target[2] = forward;
	target[3] = {0, 0, 0, 1};
	target[0].m128_f32[3] = position.GetX();
	target[1].m128_f32[3] = position.GetY();
	target[2].m128_f32[3] = position.GetZ();
	return target;
}
Math::Vector4 XM_CALLCONV cross(const Math::Vector4& v1, const Math::Vector4& v2, const Math::Vector4& v3) noexcept {
	XMVECTOR& V1 = (XMVECTOR&)v1;
	XMVECTOR& V2 = (XMVECTOR&)v2;
	XMVECTOR& V3 = (XMVECTOR&)v3;
	// V2zwyz * V3wzwy
	XMVECTOR vResult = XM_PERMUTE_PS(V2, _MM_SHUFFLE(2, 1, 3, 2));
	XMVECTOR vTemp3 = XM_PERMUTE_PS(V3, _MM_SHUFFLE(1, 3, 2, 3));
	vResult = _mm_mul_ps(vResult, vTemp3);
	// - V2wzwy * V3zwyz
	XMVECTOR vTemp2 = XM_PERMUTE_PS(V2, _MM_SHUFFLE(1, 3, 2, 3));
	vTemp3 = XM_PERMUTE_PS(vTemp3, _MM_SHUFFLE(1, 3, 0, 1));
	vTemp2 = _mm_mul_ps(vTemp2, vTemp3);
	vResult = _mm_sub_ps(vResult, vTemp2);
	// term1 * V1yxxx
	XMVECTOR vTemp1 = XM_PERMUTE_PS(V1, _MM_SHUFFLE(0, 0, 0, 1));
	vResult = _mm_mul_ps(vResult, vTemp1);
	// V2ywxz * V3wxwx
	vTemp2 = XM_PERMUTE_PS(V2, _MM_SHUFFLE(2, 0, 3, 1));
	vTemp3 = XM_PERMUTE_PS(V3, _MM_SHUFFLE(0, 3, 0, 3));
	vTemp3 = _mm_mul_ps(vTemp3, vTemp2);
	// - V2wxwx * V3ywxz
	vTemp2 = XM_PERMUTE_PS(vTemp2, _MM_SHUFFLE(2, 1, 2, 1));
	vTemp1 = XM_PERMUTE_PS(V3, _MM_SHUFFLE(2, 0, 3, 1));
	vTemp2 = _mm_mul_ps(vTemp2, vTemp1);
	vTemp3 = _mm_sub_ps(vTemp3, vTemp2);
	// vResult - temp * V1zzyy
	vTemp1 = XM_PERMUTE_PS(V1, _MM_SHUFFLE(1, 1, 2, 2));
	vTemp1 = _mm_mul_ps(vTemp1, vTemp3);
	vResult = _mm_sub_ps(vResult, vTemp1);
	// V2yzxy * V3zxyx
	vTemp2 = XM_PERMUTE_PS(V2, _MM_SHUFFLE(1, 0, 2, 1));
	vTemp3 = XM_PERMUTE_PS(V3, _MM_SHUFFLE(0, 1, 0, 2));
	vTemp3 = _mm_mul_ps(vTemp3, vTemp2);
	// - V2zxyx * V3yzxy
	vTemp2 = XM_PERMUTE_PS(vTemp2, _MM_SHUFFLE(2, 0, 2, 1));
	vTemp1 = XM_PERMUTE_PS(V3, _MM_SHUFFLE(1, 0, 2, 1));
	vTemp1 = _mm_mul_ps(vTemp1, vTemp2);
	vTemp3 = _mm_sub_ps(vTemp3, vTemp1);
	// vResult + term * V1wwwz
	vTemp1 = XM_PERMUTE_PS(V1, _MM_SHUFFLE(2, 3, 3, 3));
	vTemp3 = _mm_mul_ps(vTemp3, vTemp1);
	vResult = _mm_add_ps(vResult, vTemp3);
	return vResult;
}
Math::Vector3 XM_CALLCONV cross(const Math::Vector3& v1, const Math::Vector3& v2) noexcept {
	XMVECTOR& V1 = (XMVECTOR&)v1;
	XMVECTOR& V2 = (XMVECTOR&)v2;
	XMVECTOR vTemp1 = XM_PERMUTE_PS(V1, _MM_SHUFFLE(3, 0, 2, 1));
	// z2,x2,y2,w2
	XMVECTOR vTemp2 = XM_PERMUTE_PS(V2, _MM_SHUFFLE(3, 1, 0, 2));
	// Perform the left operation
	XMVECTOR vResult = _mm_mul_ps(vTemp1, vTemp2);
	// z1,x1,y1,w1
	vTemp1 = XM_PERMUTE_PS(vTemp1, _MM_SHUFFLE(3, 0, 2, 1));
	// y2,z2,x2,w2
	vTemp2 = XM_PERMUTE_PS(vTemp2, _MM_SHUFFLE(3, 1, 0, 2));
	// Perform the right operation
	vTemp1 = _mm_mul_ps(vTemp1, vTemp2);
	// Subract the right from left, and return answer
	vResult = _mm_sub_ps(vResult, vTemp1);
	// Set w to zero
	return _mm_and_ps(vResult, g_XMMask3);
}
Math::Vector4 XM_CALLCONV normalize(const Math::Vector4& v) noexcept {
	XMVECTOR& V = (XMVECTOR&)v;
	XMVECTOR vLengthSq = _mm_mul_ps(V, V);
	// vTemp has z and w
	XMVECTOR vTemp = XM_PERMUTE_PS(vLengthSq, _MM_SHUFFLE(3, 2, 3, 2));
	// x+z, y+w
	vLengthSq = _mm_add_ps(vLengthSq, vTemp);
	// x+z,x+z,x+z,y+w
	vLengthSq = XM_PERMUTE_PS(vLengthSq, _MM_SHUFFLE(1, 0, 0, 0));
	// ??,??,y+w,y+w
	vTemp = _mm_shuffle_ps(vTemp, vLengthSq, _MM_SHUFFLE(3, 3, 0, 0));
	// ??,??,x+z+y+w,??
	vLengthSq = _mm_add_ps(vLengthSq, vTemp);
	// Splat the length
	vLengthSq = XM_PERMUTE_PS(vLengthSq, _MM_SHUFFLE(2, 2, 2, 2));
	// Prepare for the division
	XMVECTOR vResult = _mm_sqrt_ps(vLengthSq);
	// Create zero with a single instruction
	XMVECTOR vZeroMask = _mm_setzero_ps();
	// Test for a divide by zero (Must be FP to detect -0.0)
	vZeroMask = _mm_cmpneq_ps(vZeroMask, vResult);
	// Failsafe on zero (Or epsilon) length planes
	// If the length is infinity, set the elements to zero
	vLengthSq = _mm_cmpneq_ps(vLengthSq, g_XMInfinity);
	// Divide to perform the normalization
	vResult = _mm_div_ps(V, vResult);
	// Any that are infinity, set to zero
	vResult = _mm_and_ps(vResult, vZeroMask);
	// Select qnan or result based on infinite length
	XMVECTOR vTemp1 = _mm_andnot_ps(vLengthSq, g_XMQNaN);
	XMVECTOR vTemp2 = _mm_and_ps(vResult, vLengthSq);
	vResult = _mm_or_ps(vTemp1, vTemp2);
	return vResult;
}
Math::Vector3 XM_CALLCONV normalize(const Math::Vector3& v) noexcept {
	XMVECTOR& V = (XMVECTOR&)v;
	// Perform the dot product on x,y and z only
	XMVECTOR vLengthSq = _mm_mul_ps(V, V);
	XMVECTOR vTemp = XM_PERMUTE_PS(vLengthSq, _MM_SHUFFLE(2, 1, 2, 1));
	vLengthSq = _mm_add_ss(vLengthSq, vTemp);
	vTemp = XM_PERMUTE_PS(vTemp, _MM_SHUFFLE(1, 1, 1, 1));
	vLengthSq = _mm_add_ss(vLengthSq, vTemp);
	vLengthSq = XM_PERMUTE_PS(vLengthSq, _MM_SHUFFLE(0, 0, 0, 0));
	// Prepare for the division
	XMVECTOR vResult = _mm_sqrt_ps(vLengthSq);
	// Create zero with a single instruction
	XMVECTOR vZeroMask = _mm_setzero_ps();
	// Test for a divide by zero (Must be FP to detect -0.0)
	vZeroMask = _mm_cmpneq_ps(vZeroMask, vResult);
	// Failsafe on zero (Or epsilon) length planes
	// If the length is infinity, set the elements to zero
	vLengthSq = _mm_cmpneq_ps(vLengthSq, g_XMInfinity);
	// Divide to perform the normalization
	vResult = _mm_div_ps(V, vResult);
	// Any that are infinity, set to zero
	vResult = _mm_and_ps(vResult, vZeroMask);
	// Select qnan or result based on infinite length
	XMVECTOR vTemp1 = _mm_andnot_ps(vLengthSq, g_XMQNaN);
	XMVECTOR vTemp2 = _mm_and_ps(vResult, vLengthSq);
	vResult = _mm_or_ps(vTemp1, vTemp2);
	return vResult;
}
Math::Matrix4 XM_CALLCONV GetInverseTransformMatrix(const Math::Vector3& right, const Math::Vector3& up, const Math::Vector3& forward, const Math::Vector3& position) noexcept {
	// Keep camera's axes orthogonal to each other and of unit length.
	Math::Vector3 L = normalize(forward);
	Math::Vector3 U = normalize(cross(L, right));
	// U, L already ortho-normal, so no need to normalize cross product.
	Math::Vector3 R = cross(U, L);
	// Fill in the view matrix entries.
	float x = -dot(position, R);
	float y = -dot(position, U);
	float z = -dot(position, L);
	float4x4 mView;
	mView(0, 0) = right.GetX();
	mView(1, 0) = right.GetY();
	mView(2, 0) = right.GetZ();
	mView(3, 0) = x;
	mView(0, 1) = up.GetX();
	mView(1, 1) = up.GetY();
	mView(2, 1) = up.GetZ();
	mView(3, 1) = y;
	mView(0, 2) = forward.GetX();
	mView(1, 2) = forward.GetY();
	mView(2, 2) = forward.GetZ();
	mView(3, 2) = z;
	mView(0, 3) = 0.0f;
	mView(1, 3) = 0.0f;
	mView(2, 3) = 0.0f;
	mView(3, 3) = 1.0f;
	return mView;
}
Math::Matrix4 XM_CALLCONV inverse(const Math::Matrix4& m) noexcept {
	Math::Matrix4 MT = transpose(m);
	XMVECTOR V00 = XM_PERMUTE_PS(MT[2], _MM_SHUFFLE(1, 1, 0, 0));
	XMVECTOR V10 = XM_PERMUTE_PS(MT[3], _MM_SHUFFLE(3, 2, 3, 2));
	XMVECTOR V01 = XM_PERMUTE_PS(MT[0], _MM_SHUFFLE(1, 1, 0, 0));
	XMVECTOR V11 = XM_PERMUTE_PS(MT[1], _MM_SHUFFLE(3, 2, 3, 2));
	XMVECTOR V02 = _mm_shuffle_ps(MT[2], MT[0], _MM_SHUFFLE(2, 0, 2, 0));
	XMVECTOR V12 = _mm_shuffle_ps(MT[3], MT[1], _MM_SHUFFLE(3, 1, 3, 1));
	XMVECTOR D0 = _mm_mul_ps(V00, V10);
	XMVECTOR D1 = _mm_mul_ps(V01, V11);
	XMVECTOR D2 = _mm_mul_ps(V02, V12);
	V00 = XM_PERMUTE_PS(MT[2], _MM_SHUFFLE(3, 2, 3, 2));
	V10 = XM_PERMUTE_PS(MT[3], _MM_SHUFFLE(1, 1, 0, 0));
	V01 = XM_PERMUTE_PS(MT[0], _MM_SHUFFLE(3, 2, 3, 2));
	V11 = XM_PERMUTE_PS(MT[1], _MM_SHUFFLE(1, 1, 0, 0));
	V02 = _mm_shuffle_ps(MT[2], MT[0], _MM_SHUFFLE(3, 1, 3, 1));
	V12 = _mm_shuffle_ps(MT[3], MT[1], _MM_SHUFFLE(2, 0, 2, 0));
	V00 = _mm_mul_ps(V00, V10);
	V01 = _mm_mul_ps(V01, V11);
	V02 = _mm_mul_ps(V02, V12);
	D0 = _mm_sub_ps(D0, V00);
	D1 = _mm_sub_ps(D1, V01);
	D2 = _mm_sub_ps(D2, V02);
	// V11 = D0Y,D0W,D2Y,D2Y
	V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 1, 3, 1));
	V00 = XM_PERMUTE_PS(MT[1], _MM_SHUFFLE(1, 0, 2, 1));
	V10 = _mm_shuffle_ps(V11, D0, _MM_SHUFFLE(0, 3, 0, 2));
	V01 = XM_PERMUTE_PS(MT[0], _MM_SHUFFLE(0, 1, 0, 2));
	V11 = _mm_shuffle_ps(V11, D0, _MM_SHUFFLE(2, 1, 2, 1));
	// V13 = D1Y,D1W,D2W,D2W
	XMVECTOR V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 3, 3, 1));
	V02 = XM_PERMUTE_PS(MT[3], _MM_SHUFFLE(1, 0, 2, 1));
	V12 = _mm_shuffle_ps(V13, D1, _MM_SHUFFLE(0, 3, 0, 2));
	XMVECTOR V03 = XM_PERMUTE_PS(MT[2], _MM_SHUFFLE(0, 1, 0, 2));
	V13 = _mm_shuffle_ps(V13, D1, _MM_SHUFFLE(2, 1, 2, 1));
	XMVECTOR C0 = _mm_mul_ps(V00, V10);
	XMVECTOR C2 = _mm_mul_ps(V01, V11);
	XMVECTOR C4 = _mm_mul_ps(V02, V12);
	XMVECTOR C6 = _mm_mul_ps(V03, V13);
	// V11 = D0X,D0Y,D2X,D2X
	V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(0, 0, 1, 0));
	V00 = XM_PERMUTE_PS(MT[1], _MM_SHUFFLE(2, 1, 3, 2));
	V10 = _mm_shuffle_ps(D0, V11, _MM_SHUFFLE(2, 1, 0, 3));
	V01 = XM_PERMUTE_PS(MT[0], _MM_SHUFFLE(1, 3, 2, 3));
	V11 = _mm_shuffle_ps(D0, V11, _MM_SHUFFLE(0, 2, 1, 2));
	// V13 = D1X,D1Y,D2Z,D2Z
	V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(2, 2, 1, 0));
	V02 = XM_PERMUTE_PS(MT[3], _MM_SHUFFLE(2, 1, 3, 2));
	V12 = _mm_shuffle_ps(D1, V13, _MM_SHUFFLE(2, 1, 0, 3));
	V03 = XM_PERMUTE_PS(MT[2], _MM_SHUFFLE(1, 3, 2, 3));
	V13 = _mm_shuffle_ps(D1, V13, _MM_SHUFFLE(0, 2, 1, 2));
	V00 = _mm_mul_ps(V00, V10);
	V01 = _mm_mul_ps(V01, V11);
	V02 = _mm_mul_ps(V02, V12);
	V03 = _mm_mul_ps(V03, V13);
	C0 = _mm_sub_ps(C0, V00);
	C2 = _mm_sub_ps(C2, V01);
	C4 = _mm_sub_ps(C4, V02);
	C6 = _mm_sub_ps(C6, V03);
	V00 = XM_PERMUTE_PS(MT[1], _MM_SHUFFLE(0, 3, 0, 3));
	// V10 = D0Z,D0Z,D2X,D2Y
	V10 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 0, 2, 2));
	V10 = XM_PERMUTE_PS(V10, _MM_SHUFFLE(0, 2, 3, 0));
	V01 = XM_PERMUTE_PS(MT[0], _MM_SHUFFLE(2, 0, 3, 1));
	// V11 = D0X,D0W,D2X,D2Y
	V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 0, 3, 0));
	V11 = XM_PERMUTE_PS(V11, _MM_SHUFFLE(2, 1, 0, 3));
	V02 = XM_PERMUTE_PS(MT[3], _MM_SHUFFLE(0, 3, 0, 3));
	// V12 = D1Z,D1Z,D2Z,D2W
	V12 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 2, 2, 2));
	V12 = XM_PERMUTE_PS(V12, _MM_SHUFFLE(0, 2, 3, 0));
	V03 = XM_PERMUTE_PS(MT[2], _MM_SHUFFLE(2, 0, 3, 1));
	// V13 = D1X,D1W,D2Z,D2W
	V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 2, 3, 0));
	V13 = XM_PERMUTE_PS(V13, _MM_SHUFFLE(2, 1, 0, 3));
	V00 = _mm_mul_ps(V00, V10);
	V01 = _mm_mul_ps(V01, V11);
	V02 = _mm_mul_ps(V02, V12);
	V03 = _mm_mul_ps(V03, V13);
	XMVECTOR C1 = _mm_sub_ps(C0, V00);
	C0 = _mm_add_ps(C0, V00);
	XMVECTOR C3 = _mm_add_ps(C2, V01);
	C2 = _mm_sub_ps(C2, V01);
	XMVECTOR C5 = _mm_sub_ps(C4, V02);
	C4 = _mm_add_ps(C4, V02);
	XMVECTOR C7 = _mm_add_ps(C6, V03);
	C6 = _mm_sub_ps(C6, V03);
	C0 = _mm_shuffle_ps(C0, C1, _MM_SHUFFLE(3, 1, 2, 0));
	C2 = _mm_shuffle_ps(C2, C3, _MM_SHUFFLE(3, 1, 2, 0));
	C4 = _mm_shuffle_ps(C4, C5, _MM_SHUFFLE(3, 1, 2, 0));
	C6 = _mm_shuffle_ps(C6, C7, _MM_SHUFFLE(3, 1, 2, 0));
	C0 = XM_PERMUTE_PS(C0, _MM_SHUFFLE(3, 1, 2, 0));
	C2 = XM_PERMUTE_PS(C2, _MM_SHUFFLE(3, 1, 2, 0));
	C4 = XM_PERMUTE_PS(C4, _MM_SHUFFLE(3, 1, 2, 0));
	C6 = XM_PERMUTE_PS(C6, _MM_SHUFFLE(3, 1, 2, 0));
	// Get the determinate
	XMVECTOR vTemp = XMVector4Dot(C0, MT[0]);
	vTemp = _mm_div_ps(g_XMOne, vTemp);
	XMMATRIX mResult;
	mResult.r[0] = _mm_mul_ps(C0, vTemp);
	mResult.r[1] = _mm_mul_ps(C2, vTemp);
	mResult.r[2] = _mm_mul_ps(C4, vTemp);
	mResult.r[3] = _mm_mul_ps(C6, vTemp);
	return mResult;
}
Math::Matrix3 XM_CALLCONV transpose(const Math::Matrix3& m) noexcept {
	XMMATRIX& M = (XMMATRIX&)m;
	// x.x,x.y,y.x,y.y
	XMVECTOR vTemp1 = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(1, 0, 1, 0));
	// x.z,x.w,y.z,y.w
	XMVECTOR vTemp3 = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(3, 2, 3, 2));
	// z.x,z.y,w.x,w.y
	XMVECTOR vTemp2 = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(1, 0, 1, 0));
	// z.z,z.w,w.z,w.w
	XMVECTOR vTemp4 = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(3, 2, 3, 2));
	Math::Matrix3 mResult;
	// x.x,y.x,z.x,w.x
	mResult[0] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(2, 0, 2, 0));
	// x.y,y.y,z.y,w.y
	mResult[1] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(3, 1, 3, 1));
	// x.z,y.z,z.z,w.z
	mResult[2] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(2, 0, 2, 0));
	return mResult;
}
Math::Matrix4 XM_CALLCONV transpose(const Math::Matrix4& m) noexcept {
	XMMATRIX& M = (XMMATRIX&)m;
	// x.x,x.y,y.x,y.y
	XMVECTOR vTemp1 = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(1, 0, 1, 0));
	// x.z,x.w,y.z,y.w
	XMVECTOR vTemp3 = _mm_shuffle_ps(M.r[0], M.r[1], _MM_SHUFFLE(3, 2, 3, 2));
	// z.x,z.y,w.x,w.y
	XMVECTOR vTemp2 = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(1, 0, 1, 0));
	// z.z,z.w,w.z,w.w
	XMVECTOR vTemp4 = _mm_shuffle_ps(M.r[2], M.r[3], _MM_SHUFFLE(3, 2, 3, 2));
	Math::Matrix4 mResult;
	// x.x,y.x,z.x,w.x
	mResult[0] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(2, 0, 2, 0));
	// x.y,y.y,z.y,w.y
	mResult[1] = _mm_shuffle_ps(vTemp1, vTemp2, _MM_SHUFFLE(3, 1, 3, 1));
	// x.z,y.z,z.z,w.z
	mResult[2] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(2, 0, 2, 0));
	// x.w,y.w,z.w,w.w
	mResult[3] = _mm_shuffle_ps(vTemp3, vTemp4, _MM_SHUFFLE(3, 1, 3, 1));
	return mResult;
}
Math::Matrix4 XM_CALLCONV mul(
	const Math::Matrix4& m1,
	const Math::Matrix4& m2) noexcept {
	const XMMATRIX& M1 = (const XMMATRIX&)m1;
	const XMMATRIX& M2 = (const XMMATRIX&)m2;
#if defined(_XM_NO_INTRINSICS_)
	XMMATRIX mResult;
	// Cache the invariants in registers
	float x = M1.m[0][0];
	float y = M1.m[0][1];
	float z = M1.m[0][2];
	float w = M1.m[0][3];
	// Perform the operation on the first row
	mResult.m[0][0] = (M2.m[0][0] * x) + (M2.m[1][0] * y) + (M2.m[2][0] * z) + (M2.m[3][0] * w);
	mResult.m[0][1] = (M2.m[0][1] * x) + (M2.m[1][1] * y) + (M2.m[2][1] * z) + (M2.m[3][1] * w);
	mResult.m[0][2] = (M2.m[0][2] * x) + (M2.m[1][2] * y) + (M2.m[2][2] * z) + (M2.m[3][2] * w);
	mResult.m[0][3] = (M2.m[0][3] * x) + (M2.m[1][3] * y) + (M2.m[2][3] * z) + (M2.m[3][3] * w);
	// Repeat for all the other rows
	x = M1.m[1][0];
	y = M1.m[1][1];
	z = M1.m[1][2];
	w = M1.m[1][3];
	mResult.m[1][0] = (M2.m[0][0] * x) + (M2.m[1][0] * y) + (M2.m[2][0] * z) + (M2.m[3][0] * w);
	mResult.m[1][1] = (M2.m[0][1] * x) + (M2.m[1][1] * y) + (M2.m[2][1] * z) + (M2.m[3][1] * w);
	mResult.m[1][2] = (M2.m[0][2] * x) + (M2.m[1][2] * y) + (M2.m[2][2] * z) + (M2.m[3][2] * w);
	mResult.m[1][3] = (M2.m[0][3] * x) + (M2.m[1][3] * y) + (M2.m[2][3] * z) + (M2.m[3][3] * w);
	x = M1.m[2][0];
	y = M1.m[2][1];
	z = M1.m[2][2];
	w = M1.m[2][3];
	mResult.m[2][0] = (M2.m[0][0] * x) + (M2.m[1][0] * y) + (M2.m[2][0] * z) + (M2.m[3][0] * w);
	mResult.m[2][1] = (M2.m[0][1] * x) + (M2.m[1][1] * y) + (M2.m[2][1] * z) + (M2.m[3][1] * w);
	mResult.m[2][2] = (M2.m[0][2] * x) + (M2.m[1][2] * y) + (M2.m[2][2] * z) + (M2.m[3][2] * w);
	mResult.m[2][3] = (M2.m[0][3] * x) + (M2.m[1][3] * y) + (M2.m[2][3] * z) + (M2.m[3][3] * w);
	x = M1.m[3][0];
	y = M1.m[3][1];
	z = M1.m[3][2];
	w = M1.m[3][3];
	mResult.m[3][0] = (M2.m[0][0] * x) + (M2.m[1][0] * y) + (M2.m[2][0] * z) + (M2.m[3][0] * w);
	mResult.m[3][1] = (M2.m[0][1] * x) + (M2.m[1][1] * y) + (M2.m[2][1] * z) + (M2.m[3][1] * w);
	mResult.m[3][2] = (M2.m[0][2] * x) + (M2.m[1][2] * y) + (M2.m[2][2] * z) + (M2.m[3][2] * w);
	mResult.m[3][3] = (M2.m[0][3] * x) + (M2.m[1][3] * y) + (M2.m[2][3] * z) + (M2.m[3][3] * w);
	return mResult;
#elif defined(_XM_ARM_NEON_INTRINSICS_)
	XMMATRIX mResult;
	float32x2_t VL = vget_low_f32(M1.r[0]);
	float32x2_t VH = vget_high_f32(M1.r[0]);
	// Perform the operation on the first row
	XMVECTOR vX = vmulq_lane_f32(M2.r[0], VL, 0);
	XMVECTOR vY = vmulq_lane_f32(M2.r[1], VL, 1);
	XMVECTOR vZ = vmlaq_lane_f32(vX, M2.r[2], VH, 0);
	XMVECTOR vW = vmlaq_lane_f32(vY, M2.r[3], VH, 1);
	mResult.r[0] = vaddq_f32(vZ, vW);
	// Repeat for the other 3 rows
	VL = vget_low_f32(M1.r[1]);
	VH = vget_high_f32(M1.r[1]);
	vX = vmulq_lane_f32(M2.r[0], VL, 0);
	vY = vmulq_lane_f32(M2.r[1], VL, 1);
	vZ = vmlaq_lane_f32(vX, M2.r[2], VH, 0);
	vW = vmlaq_lane_f32(vY, M2.r[3], VH, 1);
	mResult.r[1] = vaddq_f32(vZ, vW);
	VL = vget_low_f32(M1.r[2]);
	VH = vget_high_f32(M1.r[2]);
	vX = vmulq_lane_f32(M2.r[0], VL, 0);
	vY = vmulq_lane_f32(M2.r[1], VL, 1);
	vZ = vmlaq_lane_f32(vX, M2.r[2], VH, 0);
	vW = vmlaq_lane_f32(vY, M2.r[3], VH, 1);
	mResult.r[2] = vaddq_f32(vZ, vW);
	VL = vget_low_f32(M1.r[3]);
	VH = vget_high_f32(M1.r[3]);
	vX = vmulq_lane_f32(M2.r[0], VL, 0);
	vY = vmulq_lane_f32(M2.r[1], VL, 1);
	vZ = vmlaq_lane_f32(vX, M2.r[2], VH, 0);
	vW = vmlaq_lane_f32(vY, M2.r[3], VH, 1);
	mResult.r[3] = vaddq_f32(vZ, vW);
	return mResult;
#elif defined(_XM_SSE_INTRINSICS_)
	XMMATRIX mResult;
	// Splat the component X,Y,Z then W
#if defined(_XM_AVX_INTRINSICS_)
	XMVECTOR vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 0);
	XMVECTOR vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 1);
	XMVECTOR vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 2);
	XMVECTOR vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[0]) + 3);
#else
	// Use vW to hold the original row
	XMVECTOR vW = M1.r[0];
	XMVECTOR vX = XM_PERMUTE_PS(vW, _MM_SHUFFLE(0, 0, 0, 0));
	XMVECTOR vY = XM_PERMUTE_PS(vW, _MM_SHUFFLE(1, 1, 1, 1));
	XMVECTOR vZ = XM_PERMUTE_PS(vW, _MM_SHUFFLE(2, 2, 2, 2));
	vW = XM_PERMUTE_PS(vW, _MM_SHUFFLE(3, 3, 3, 3));
#endif
	// Perform the operation on the first row
	vX = _mm_mul_ps(vX, M2.r[0]);
	vY = _mm_mul_ps(vY, M2.r[1]);
	vZ = _mm_mul_ps(vZ, M2.r[2]);
	vW = _mm_mul_ps(vW, M2.r[3]);
	// Perform a binary add to reduce cumulative errors
	vX = _mm_add_ps(vX, vZ);
	vY = _mm_add_ps(vY, vW);
	vX = _mm_add_ps(vX, vY);
	mResult.r[0] = vX;
	// Repeat for the other 3 rows
#if defined(_XM_AVX_INTRINSICS_)
	vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 0);
	vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 1);
	vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 2);
	vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[1]) + 3);
#else
	vW = M1.r[1];
	vX = XM_PERMUTE_PS(vW, _MM_SHUFFLE(0, 0, 0, 0));
	vY = XM_PERMUTE_PS(vW, _MM_SHUFFLE(1, 1, 1, 1));
	vZ = XM_PERMUTE_PS(vW, _MM_SHUFFLE(2, 2, 2, 2));
	vW = XM_PERMUTE_PS(vW, _MM_SHUFFLE(3, 3, 3, 3));
#endif
	vX = _mm_mul_ps(vX, M2.r[0]);
	vY = _mm_mul_ps(vY, M2.r[1]);
	vZ = _mm_mul_ps(vZ, M2.r[2]);
	vW = _mm_mul_ps(vW, M2.r[3]);
	vX = _mm_add_ps(vX, vZ);
	vY = _mm_add_ps(vY, vW);
	vX = _mm_add_ps(vX, vY);
	mResult.r[1] = vX;
#if defined(_XM_AVX_INTRINSICS_)
	vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 0);
	vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 1);
	vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 2);
	vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[2]) + 3);
#else
	vW = M1.r[2];
	vX = XM_PERMUTE_PS(vW, _MM_SHUFFLE(0, 0, 0, 0));
	vY = XM_PERMUTE_PS(vW, _MM_SHUFFLE(1, 1, 1, 1));
	vZ = XM_PERMUTE_PS(vW, _MM_SHUFFLE(2, 2, 2, 2));
	vW = XM_PERMUTE_PS(vW, _MM_SHUFFLE(3, 3, 3, 3));
#endif
	vX = _mm_mul_ps(vX, M2.r[0]);
	vY = _mm_mul_ps(vY, M2.r[1]);
	vZ = _mm_mul_ps(vZ, M2.r[2]);
	vW = _mm_mul_ps(vW, M2.r[3]);
	vX = _mm_add_ps(vX, vZ);
	vY = _mm_add_ps(vY, vW);
	vX = _mm_add_ps(vX, vY);
	mResult.r[2] = vX;
#if defined(_XM_AVX_INTRINSICS_)
	vX = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 0);
	vY = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 1);
	vZ = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 2);
	vW = _mm_broadcast_ss(reinterpret_cast<const float*>(&M1.r[3]) + 3);
#else
	vW = M1.r[3];
	vX = XM_PERMUTE_PS(vW, _MM_SHUFFLE(0, 0, 0, 0));
	vY = XM_PERMUTE_PS(vW, _MM_SHUFFLE(1, 1, 1, 1));
	vZ = XM_PERMUTE_PS(vW, _MM_SHUFFLE(2, 2, 2, 2));
	vW = XM_PERMUTE_PS(vW, _MM_SHUFFLE(3, 3, 3, 3));
#endif
	vX = _mm_mul_ps(vX, M2.r[0]);
	vY = _mm_mul_ps(vY, M2.r[1]);
	vZ = _mm_mul_ps(vZ, M2.r[2]);
	vW = _mm_mul_ps(vW, M2.r[3]);
	vX = _mm_add_ps(vX, vZ);
	vY = _mm_add_ps(vY, vW);
	vX = _mm_add_ps(vX, vY);
	mResult.r[3] = vX;
	return mResult;
#endif
}
