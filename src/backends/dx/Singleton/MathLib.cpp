#include <Singleton/MathLib.h>
#include <vstl/MetaLib.h>
using namespace Math;
#define GetVec(name, v) Vector4 name = XMLoadFloat3(&##v);
#define StoreVec(ptr, v) XMStoreFloat3(ptr, v);
Vector4 MathLib::GetPlane(
	const Vector3& a,
	const Vector3& b,
	const Vector3& c) noexcept {
	Vector3 normal = normalize(cross(b - a, c - a));
	float disVec = -dot(normal, a);
	Vector4& v = reinterpret_cast<Vector4&>(normal);
	v.SetW(disVec);
	return v;
}
float MathLib::GetScreenMultiple(
	float fov,
	float aspect) noexcept {
	float tanFov = tan(0.5f * fov);
	return Max(
			   (1 / tanFov),
			   aspect / tanFov)
		   * 0.5f;
}
float MathLib::CalculateTextureMipQFactor(
	float fov,
	float distance,
	uint camRes,
	float texRes,
	float panelSize) noexcept {
	float v = camRes / (distance * tan(fov * 0.5f) * 2);
	float size = (texRes / panelSize) / v;
	return 0.5f * log2(size * size);
}
float MathLib::CalculateMipDistance(float fov, float mip, uint camRes, float texRes, float panelSize) noexcept {
	float r = (texRes / panelSize);
	float rDivideV = sqrt(pow(2, mip * 2));
	float v = r / rDivideV;
	v = camRes / v;
	v /= (tan(fov * 0.5f) * 2);
	return v;
}
float MathLib::GetScreenPercentage(Math::Matrix4 const& projectMat,
								   float objectToCameraDistance,
								   float sphereRadius) {
	float screenMultiple = Max(0.5f * projectMat[0].m128_f32[0], 0.5f * projectMat[1].m128_f32[1]);
	float screenRadius = screenMultiple * sphereRadius / Max(1.0f, objectToCameraDistance);
	return screenRadius * 2;
}
double MathLib::GetHalton_Float(uint index) {
	index = (index << 16) | (index >> 16);
	index = ((index & 0x00ff00ff) << 8) | ((index & 0xff00ff00) >> 8);
	index = ((index & 0x0f0f0f0f) << 4) | ((index & 0xf0f0f0f0) >> 4);
	index = ((index & 0x33333333) << 2) | ((index & 0xcccccccc) >> 2);
	index = ((index & 0x55555555) << 1) | ((index & 0xaaaaaaaa) >> 1);
	uint u = 0x3f800000u | (index >> 9);
	return ((float&)u) - 1;
}
double2 MathLib::GetHalton_Vector2(uint index) {
	static constexpr uint16_t m_perm3[243] = {0, 81, 162, 27, 108, 189, 54, 135, 216, 9, 90, 171, 36, 117, 198, 63, 144, 225, 18, 99, 180, 45, 126, 207, 72, 153, 234, 3, 84, 165, 30, 111, 192, 57, 138, 219, 12, 93, 174, 39, 120, 201, 66, 147, 228, 21, 102, 183, 48, 129, 210, 75, 156, 237, 6, 87, 168, 33, 114, 195, 60, 141, 222, 15, 96, 177, 42, 123, 204, 69, 150, 231, 24, 105, 186, 51, 132, 213, 78, 159, 240, 1, 82, 163, 28, 109, 190, 55, 136, 217, 10, 91, 172, 37, 118, 199, 64, 145, 226, 19, 100, 181, 46, 127, 208, 73, 154, 235, 4, 85, 166, 31, 112, 193, 58, 139, 220, 13, 94, 175, 40, 121, 202, 67, 148, 229, 22, 103, 184, 49, 130, 211, 76, 157, 238, 7, 88, 169, 34, 115, 196, 61, 142, 223, 16, 97, 178, 43, 124, 205, 70, 151, 232, 25, 106, 187, 52, 133, 214, 79, 160, 241, 2, 83, 164, 29, 110, 191, 56, 137, 218, 11, 92, 173, 38, 119, 200, 65, 146, 227, 20, 101, 182, 47, 128, 209, 74, 155, 236, 5, 86, 167, 32, 113, 194, 59, 140, 221, 14, 95, 176, 41, 122, 203, 68, 149, 230, 23, 104, 185, 50, 131, 212, 77, 158, 239, 8, 89, 170, 35, 116, 197, 62, 143, 224, 17, 98, 179, 44, 125, 206, 71, 152, 233, 26, 107, 188, 53, 134, 215, 80, 161, 242};
	return double2(GetHalton_Float(index), (m_perm3[index % 243u] * 14348907u
											+ m_perm3[(index / 243u) % 243u] * 59049u
											+ m_perm3[(index / 59049u) % 243u] * 243u
											+ m_perm3[(index / 14348907u) % 243u])
											   * double(0x1.fffffcp-1 / 3486784401u));
}
double3 MathLib::GetHalton_Vector3(uint index) {
	static constexpr uint16_t m_perm5[125] = {0, 75, 50, 25, 100, 15, 90, 65, 40, 115, 10, 85, 60, 35, 110, 5, 80, 55, 30, 105, 20, 95, 70, 45, 120, 3, 78, 53, 28, 103, 18, 93, 68, 43, 118, 13, 88, 63, 38, 113, 8, 83, 58, 33, 108, 23, 98, 73, 48, 123, 2, 77, 52, 27, 102, 17, 92, 67, 42, 117, 12, 87, 62, 37, 112, 7, 82, 57, 32, 107, 22, 97, 72, 47, 122, 1, 76, 51, 26, 101, 16, 91, 66, 41, 116, 11, 86, 61, 36, 111, 6, 81, 56, 31, 106, 21, 96, 71, 46, 121, 4, 79, 54, 29, 104, 19, 94, 69, 44, 119, 14, 89, 64, 39, 114, 9, 84, 59, 34, 109, 24, 99, 74, 49, 124};
	double2 v = GetHalton_Vector2(index);
	return double3(v.x, v.y,
				   (m_perm5[index % 125u] * 1953125u
					+ m_perm5[(index / 125u) % 125u] * 15625u
					+ m_perm5[(index / 15625u) % 125u] * 125u
					+ m_perm5[(index / 1953125u) % 125u])
					   * double(0x1.fffffcp-1 / 244140625u));
}
float MathLib::GetScreenPercentage(
	Math::Vector3 const& boundsOrigin,
	Math::Vector3 const& camOrigin,
	float screenMultiple,
	float sphereRadius) noexcept {
	float dist = distance(boundsOrigin, camOrigin);
	float screenRadius = screenMultiple * sphereRadius / Max(1.0f, dist);
	return screenRadius;
}
Vector4 MathLib::GetPlane(
	const Vector3& normal,
	const Vector3& inPoint) noexcept {
	float dt = -dot(normal, inPoint);
	Vector4 result(normal, dt);
	return result;
}
/*
bool MathLib::BoxIntersect(const Matrix4& localToWorldMatrix, Vector4* planes, const Vector3& position, const Vector3& localExtent)
{
	Vector4 pos = mul(localToWorldMatrix, position);
	((Vector4&)(localExtent)).SetW(0);
	auto func = [&](uint i)->bool
	{
		Vector4 plane = planes[i];
		plane.SetW(0);
		Vector4 absNormal = abs(mul(localToWorldMatrix, plane));
		float result = dot(pos, plane) - dot(absNormal, (Vector4&)localExtent);
		if (result > -plane.GetW()) return false;
	};
	return InnerLoopEarlyBreak<decltype(func), 6>(func);
}*/
bool MathLib::BoxIntersect(const Matrix4& localToWorldMatrix, Vector4* planes, const Vector3& position, const Vector3& localExtent,
						   const Vector3& frustumMinPoint, const Vector3& frustumMaxPoint) noexcept {
	const_cast<Vector4&>(reinterpret_cast<Vector4 const&>(localExtent)).SetW(0);
	static const Vector3 arrays[8] =
		{
			Vector3(1, 1, 1),
			Vector3(1, -1, 1),
			Vector3(1, 1, -1),
			Vector3(1, -1, -1),
			Vector3(-1, 1, 1),
			Vector3(-1, -1, 1),
			Vector3(-1, 1, -1),
			Vector3(-1, -1, -1)};
	Vector4 minPos = mul(localToWorldMatrix, Vector4(position + arrays[0] * localExtent, 1));
	Vector4 maxPos = minPos;
	for (uint i = 1; i < 8; ++i) {
		Vector4 worldPos = Vector4(position + arrays[i] * localExtent, 1);
		worldPos = mul(localToWorldMatrix, worldPos);
		minPos = Min(minPos, worldPos);
		maxPos = Max(maxPos, worldPos);
	}
	auto a = GetBool3(reinterpret_cast<Vector3&>(minPos) > frustumMaxPoint) || GetBool3(reinterpret_cast<Vector3&>(maxPos) < frustumMinPoint);
	if (a.x || a.y || a.z) return false;
	Vector3 pos = static_cast<Vector3>(mul(localToWorldMatrix, position));
	Matrix3 normalLocalMat = transpose(localToWorldMatrix.Get3x3());
	for (uint i = 0; i < 6; ++i) {
		Vector4 plane = planes[i];
		Vector3 const& planeXYZ = reinterpret_cast<Vector3 const&>(plane);
		Vector3 absNormal = abs(mul(normalLocalMat, planeXYZ));
		float result = dot(pos, planeXYZ) - dot(absNormal, localExtent);
		if (result > -plane.GetW()) return false;
	}
	return true;
}
void MathLib::GetCameraNearPlanePoints(
	const Math::Vector3& right,
	const Math::Vector3& up,
	const Math::Vector3& forward,
	const Math::Vector3& position,
	double fov,
	double aspect,
	double distance,
	Math::Vector3* corners) noexcept {
	double upLength = distance * tan(fov * 0.5);
	double rightLength = upLength * aspect;
	Vector3 farPoint = position + distance * forward;
	Vector3 upVec = upLength * up;
	Vector3 rightVec = rightLength * right;
	corners[0] = farPoint - upVec - rightVec;
	corners[1] = farPoint - upVec + rightVec;
	corners[2] = farPoint + upVec - rightVec;
	corners[3] = farPoint + upVec + rightVec;
}
/*
void MathLib::GetCameraNearPlanePoints(
	const Matrix4& localToWorldMat,
	double fov,
	double aspect,
	double distance,
	Vector3* corners
)
{
	Matrix4& localToWorldMatrix = (Matrix4&)localToWorldMat;
	double upLength = distance * tan(fov * 0.5);
	double rightLength = upLength * aspect;
	Vector3 farPoint = localToWorldMatrix[3] + distance * localToWorldMatrix[2];
	Vector3 upVec = upLength * localToWorldMatrix[1];
	Vector3 rightVec = rightLength * localToWorldMatrix[0];
	corners[0] = farPoint - upVec - rightVec;
	corners[1] = farPoint - upVec + rightVec;
	corners[2] = farPoint + upVec - rightVec;
	corners[3] = farPoint + upVec + rightVec;
}
void MathLib::GetPerspFrustumPlanes(
	const Matrix4& localToWorldMat,
	double fov,
	double aspect,
	double nearPlane,
	double farPlane,
	float4* frustumPlanes
)
{
	Matrix4& localToWorldMatrix = (Matrix4&)localToWorldMat;
	Vector3 nearCorners[4];
	GetCameraNearPlanePoints((localToWorldMatrix), fov, aspect, nearPlane, nearCorners);
	*(Vector4*)frustumPlanes = GetPlane((localToWorldMatrix[2]), (localToWorldMatrix[3] + farPlane * localToWorldMatrix[2]));
	*(Vector4*)(frustumPlanes + 1) = GetPlane(-localToWorldMatrix[2], (localToWorldMatrix[3] + nearPlane * localToWorldMatrix[2]));
	*(Vector4*)(frustumPlanes + 2) = GetPlane((nearCorners[1]), (nearCorners[0]), (localToWorldMatrix[3]));
	*(Vector4*)(frustumPlanes + 3) = GetPlane((nearCorners[2]), (nearCorners[3]), (localToWorldMatrix[3]));
	*(Vector4*)(frustumPlanes + 4) = GetPlane((nearCorners[0]), (nearCorners[2]), (localToWorldMatrix[3]));
	*(Vector4*)(frustumPlanes + 5) = GetPlane((nearCorners[3]), (nearCorners[1]), (localToWorldMatrix[3]));
}*/
void MathLib::GetPerspFrustumPlanes(
	const Math::Vector3& right,
	const Math::Vector3& up,
	const Math::Vector3& forward,
	const Math::Vector3& position,
	double fov,
	double aspect,
	double nearPlane,
	double farPlane,
	Math::Vector4* frustumPlanes) noexcept {
	Vector3 nearCorners[4];
	GetCameraNearPlanePoints(right, up, forward, position, fov, aspect, nearPlane, nearCorners);
	frustumPlanes[0] = GetPlane(forward, (position + farPlane * forward));
	frustumPlanes[1] = GetPlane(-forward, (position + nearPlane * forward));
	frustumPlanes[2] = GetPlane((nearCorners[1]), (nearCorners[0]), (position));
	frustumPlanes[3] = GetPlane((nearCorners[2]), (nearCorners[3]), (position));
	frustumPlanes[4] = GetPlane((nearCorners[0]), (nearCorners[2]), (position));
	frustumPlanes[5] = GetPlane((nearCorners[3]), (nearCorners[1]), (position));
}
/*
void MathLib::GetPerspFrustumPlanes(
	const Math::Matrix4& localToWorldMat,
	double fov,
	double aspect,
	double nearPlane,
	double farPlane,
	Math::Vector4* frustumPlanes)
{
	Matrix4& localToWorldMatrix = (Matrix4&)localToWorldMat;
	Vector3 nearCorners[4];
	GetCameraNearPlanePoints((localToWorldMatrix), fov, aspect, nearPlane, nearCorners);
	*frustumPlanes = GetPlane((localToWorldMatrix[2]), (localToWorldMatrix[3] + farPlane * localToWorldMatrix[2]));
	*(frustumPlanes + 1) = GetPlane(-localToWorldMatrix[2], (localToWorldMatrix[3] + nearPlane * localToWorldMatrix[2]));
	*(frustumPlanes + 2) = GetPlane((nearCorners[1]), (nearCorners[0]), (localToWorldMatrix[3]));
	*(frustumPlanes + 3) = GetPlane((nearCorners[2]), (nearCorners[3]), (localToWorldMatrix[3]));
	*(frustumPlanes + 4) = GetPlane((nearCorners[0]), (nearCorners[2]), (localToWorldMatrix[3]));
	*(frustumPlanes + 5) = GetPlane((nearCorners[3]), (nearCorners[1]), (localToWorldMatrix[3]));
}
void MathLib::GetFrustumBoundingBox(
	const Matrix4& localToWorldMat,
	double nearWindowHeight,
	double farWindowHeight,
	double aspect,
	double nearZ,
	double farZ,
	Vector3* minValue,
	Vector3* maxValue)
{
	Matrix4& localToWorldMatrix = (Matrix4&)localToWorldMat;
	double halfNearYHeight = nearWindowHeight * 0.5;
	double halfFarYHeight = farWindowHeight * 0.5;
	double halfNearXWidth = halfNearYHeight * aspect;
	double halfFarXWidth = halfFarYHeight * aspect;
	Vector4 poses[8];
	Vector4 pos = localToWorldMatrix[3];
	Vector4 right = localToWorldMatrix[0];
	Vector4 up = localToWorldMatrix[1];
	Vector4 forward = localToWorldMatrix[2];
	poses[0] = pos + forward * nearZ - right * halfNearXWidth - up * halfNearYHeight;
	poses[1] = pos + forward * nearZ - right * halfNearXWidth + up * halfNearYHeight;
	poses[2] = pos + forward * nearZ + right * halfNearXWidth - up * halfNearYHeight;
	poses[3] = pos + forward * nearZ + right * halfNearXWidth + up * halfNearYHeight;
	poses[4] = pos + forward * farZ - right * halfFarXWidth - up * halfFarYHeight;
	poses[5] = pos + forward * farZ - right * halfFarXWidth + up * halfFarYHeight;
	poses[6] = pos + forward * farZ + right * halfFarXWidth - up * halfFarYHeight;
	poses[7] = pos + forward * farZ + right * halfFarXWidth + up * halfFarYHeight;
	*minValue = poses[7];
	*maxValue = poses[7];
	auto func = [&](uint i)->void
	{
		*minValue = Min<Vector3>(static_cast<Vector3>poses[i], *minValue);
		*maxValue = Max<Vector3>(static_cast<Vector3>poses[i], *maxValue);
	};
	InnerLoop<decltype(func), 7>(func);
}*/
void MathLib::GetFrustumBoundingBox(
	const Vector3& right,
	const Vector3& up,
	const Vector3& forward,
	const Vector3& pos,
	double nearWindowHeight,
	double farWindowHeight,
	double aspect,
	double nearZ,
	double farZ,
	Math::Vector3* minValue,
	Math::Vector3* maxValue) noexcept {
	double halfNearYHeight = nearWindowHeight * 0.5;
	double halfFarYHeight = farWindowHeight * 0.5;
	double halfNearXWidth = halfNearYHeight * aspect;
	double halfFarXWidth = halfFarYHeight * aspect;
	Vector3 poses[8];
	poses[0] = pos + forward * nearZ - right * halfNearXWidth - up * halfNearYHeight;
	poses[1] = pos + forward * nearZ - right * halfNearXWidth + up * halfNearYHeight;
	poses[2] = pos + forward * nearZ + right * halfNearXWidth - up * halfNearYHeight;
	poses[3] = pos + forward * nearZ + right * halfNearXWidth + up * halfNearYHeight;
	poses[4] = pos + forward * farZ - right * halfFarXWidth - up * halfFarYHeight;
	poses[5] = pos + forward * farZ - right * halfFarXWidth + up * halfFarYHeight;
	poses[6] = pos + forward * farZ + right * halfFarXWidth - up * halfFarYHeight;
	poses[7] = pos + forward * farZ + right * halfFarXWidth + up * halfFarYHeight;
	*minValue = poses[7];
	*maxValue = poses[7];
	for (uint i = 0; i < 7; ++i) {
		*minValue = Min<Vector3>(poses[i], *minValue);
		*maxValue = Max<Vector3>(poses[i], *maxValue);
	};
}
bool MathLib::ConeIntersect(const Cone& cone, const Vector4& plane) noexcept {
	Vector3 dir = cone.direction;
	Vector3 vertex = cone.vertex;
	Vector3 m = cross(cross(reinterpret_cast<Vector3 const&>(plane), dir), dir);
	Vector3 Q = vertex + dir * cone.height + normalize(m) * cone.radius;
	return (GetPointDistanceToPlane(plane, vertex) < 0) || (GetPointDistanceToPlane(plane, Q) < 0);
}
void MathLib::GetOrthoCamFrustumPlanes(
	const Math::Vector3& right,
	const Math::Vector3& up,
	const Math::Vector3& forward,
	const Math::Vector3& position,
	float xSize,
	float ySize,
	float nearPlane,
	float farPlane,
	Vector4* results) noexcept {
	Vector3 normals[6];
	Vector3 positions[6];
	normals[0] = up;
	positions[0] = position + up * ySize;
	normals[1] = -up;
	positions[1] = position - up * ySize;
	normals[2] = right;
	positions[2] = position + right * xSize;
	normals[3] = -right;
	positions[3] = position - right * xSize;
	normals[4] = forward;
	positions[4] = position + forward * farPlane;
	normals[5] = -forward;
	positions[5] = position + forward * nearPlane;
	for (uint i = 0; i < 6; ++i) {
		results[i] = GetPlane(normals[i], positions[i]);
	};
}
void MathLib::GetOrthoCamFrustumPoints(
	const Math::Vector3& right,
	const Math::Vector3& up,
	const Math::Vector3& forward,
	const Math::Vector3& position,
	float xSize,
	float ySize,
	float nearPlane,
	float farPlane,
	Vector3* results) noexcept {
	results[0] = position + xSize * right + ySize * up + farPlane * forward;
	results[1] = position + xSize * right + ySize * up + nearPlane * forward;
	results[2] = position + xSize * right - ySize * up + farPlane * forward;
	results[3] = position + xSize * right - ySize * up + nearPlane * forward;
	results[4] = position - xSize * right + ySize * up + farPlane * forward;
	results[5] = position - xSize * right + ySize * up + nearPlane * forward;
	results[6] = position - xSize * right - ySize * up + farPlane * forward;
	results[7] = position - xSize * right - ySize * up + nearPlane * forward;
}
float MathLib::DistanceToCube(const Math::Vector3& cubeSize, const Math::Vector3& relativePosToCube) noexcept {
	Vector3 absPos = abs(relativePosToCube);
	Vector4 plane = GetPlane(Vector3(0, 1, 0), Vector3(0, cubeSize.GetY(), 0));
	float dist = GetPointDistanceToPlane(plane, absPos);
	plane = GetPlane(Vector3(1, 0, 0), Vector3(cubeSize.GetX(), 0, 0));
	dist = Max(dist, GetPointDistanceToPlane(plane, absPos));
	plane = GetPlane(Vector3(0, 0, 1), Vector3(0, 0, cubeSize.GetZ()));
	dist = Max(dist, GetPointDistanceToPlane(plane, absPos));
	return dist;
}
double MathLib::DistanceToQuad(double size, float2 quadToTarget) noexcept {
	Vector3 quadVec = Vector3(quadToTarget.x, quadToTarget.y, 0);
	quadVec = abs(quadVec);
	double len = length(quadVec);
	quadVec /= len;
	double dotV = Max(dot(Vector3(0, 1, 0), quadVec), dot(Vector3(1, 0, 0), quadVec));
	double leftLen = size / dotV;
	return len - leftLen;
};
bool MathLib::BoxIntersect(const Math::Vector3& position, const Math::Vector3& extent, Math::Vector4* planes) noexcept {
	for (uint i = 0; i < 6; ++i) {
		Vector4& plane = planes[i];
		Vector3 const& planeXYZ = reinterpret_cast<Vector3 const&>(plane);
		Vector3 absNormal = abs(planeXYZ);
		if ((dot(position, planeXYZ) - dot(absNormal, extent)) > -(float)plane.GetW())
			return false;
	}
	return true;
}
bool MathLib::BoxIntersect(const Math::Vector3& position,
						   const Math::Vector3& extent,
						   Math::Vector4* planes,
						   const Math::Vector3& frustumMinPoint,
						   const Math::Vector3& frustumMaxPoint) noexcept {
	Vector3 minPos = position - extent;
	Vector3 maxPos = position + extent;
	auto a = GetBool3(minPos > frustumMaxPoint) || GetBool3(maxPos < frustumMinPoint);
	if (a.x || a.y || a.z) return false;
	for (uint i = 0; i < 6; ++i) {
		Vector4& plane = planes[i];
		Vector3 const& planeXYZ = reinterpret_cast<Vector3 const&>(plane);
		Vector3 absNormal = abs(planeXYZ);
		if ((dot(position, planeXYZ) - dot(absNormal, extent)) > -(float)plane.GetW())
			return false;
	}
	return true;
}
bool MathLib::BoxContactWithBox(const double3& min0, const double3& max0, const double3& min1, const double3& max1) noexcept {
	bool vx, vy, vz;
	vx = min0.x > max1.x;
	vy = min0.y > max1.y;
	vz = min0.z > max1.z;
	if (vx || vy || vz) return false;
	vx = min1.x > max0.x;
	vy = min1.y > max0.y;
	vz = min1.z > max0.z;
	if (vx || vy || vz) return false;
	return true;
}
Vector4 MathLib::CameraSpacePlane(const Matrix4& worldToCameraMatrix, const Vector3& pos, const Vector3& normal, float clipPlaneOffset) noexcept {
	Vector4 offsetPos = pos + normal * clipPlaneOffset;
	offsetPos.SetW(1);
	Vector4 cpos = mul(worldToCameraMatrix, offsetPos);
	cpos.SetW(0);
	Vector4& nor = const_cast<Vector4&>(reinterpret_cast<Vector4 const&>(normal));
	nor.SetW(0);
	Vector4 cnormal = normalize(mul(worldToCameraMatrix, nor));
	cnormal.SetW(-dot(cpos, cnormal));
	return cnormal;
}
Math::Matrix4 MathLib::CalculateObliqueMatrix(const Math::Vector4& clipPlane, const Math::Matrix4& projection) noexcept {
	Matrix4 inversion = inverse(projection);
	Vector4 q = mul(inversion, Vector4(
								   (clipPlane.GetX() > 0) - (clipPlane.GetX() < 0),
								   (clipPlane.GetY() > 0) - (clipPlane.GetY() < 0),
								   1.0f,
								   1.0f));
	Vector4 c = clipPlane * (2.0f / dot(clipPlane, q));
	Matrix4 retValue = projection;
	retValue[2] = Vector4(c) - Vector4(retValue[3]);
	return retValue;
}
Math::Matrix4 MathLib::GetOrthoMatrix(
	float nearPlane,
	float farPlane,
	float size,
	float aspect,
	bool renderTarget,
	bool gpuResource) noexcept {
	size *= 2;
	Matrix4 mat = XMMatrixOrthographicLH(size * aspect, size, farPlane, nearPlane);
	if (renderTarget) {
		mat[1].m128_f32[1] *= -1;
	}
	if (gpuResource)
		return mat;
	else
		return transpose(mat);
}
void MathLib::GetCubemapRenderData(
	const Math::Vector3& position,
	float nearZ,
	float farZ,
	Math::Matrix4* viewProj,
	Vector4** frustumPlanes,
	std::pair<Math::Vector3, Math::Vector3>* minMaxBound) noexcept {
	//Right
	//Left
	//Up
	//Down
	//Forward
	//Back
	Vector3 rightDirs[6] =
		{
			{0, 0, -1},
			{0, 0, 1},
			{1, 0, 0},
			{1, 0, 0},
			{1, 0, 0},
			{-1, 0, 0}};
	Vector3 upDirs[6] =
		{
			{0, 1, 0},
			{0, 1, 0},
			{0, 0, -1},
			{0, 0, 1},
			{0, 1, 0},
			{0, 1, 0}};
	Vector3 forwardDirs[6]{
		{1, 0, 0},
		{-1, 0, 0},
		{0, 1, 0},
		{0, -1, 0},
		{0, 0, 1},
		{0, 0, -1}};
	Matrix4 proj = XMMatrixPerspectiveFovLH(90 * Deg2Rad, 1, farZ, nearZ);
	double nearWindowHeight = (double)nearZ * tan(45 * Deg2Rad);
	double farWindowHeight = (double)farZ * tan(45 * Deg2Rad);
	for (uint i = 0; i < 6; ++i) {
		Vector4* frustumPlanesResult = frustumPlanes[i];
		Matrix4 view = GetInverseTransformMatrix(rightDirs[i], upDirs[i], forwardDirs[i], position);
		viewProj[i] = mul(view, proj);
		GetPerspFrustumPlanes(rightDirs[i], upDirs[i], forwardDirs[i], position, 90 * Deg2Rad, 1, nearZ, farZ, frustumPlanesResult);
		Vector3 nearCenter = position + nearZ * forwardDirs[i];
		Vector3 farCenter = position + farZ * forwardDirs[i];
		Vector3 frustumPoints[8] =
			{
				nearCenter + nearWindowHeight * upDirs[i] + nearWindowHeight * rightDirs[i],
				nearCenter + nearWindowHeight * upDirs[i] - nearWindowHeight * rightDirs[i],
				nearCenter - nearWindowHeight * upDirs[i] + nearWindowHeight * rightDirs[i],
				nearCenter - nearWindowHeight * upDirs[i] - nearWindowHeight * rightDirs[i],
				farCenter + farWindowHeight * upDirs[i] + farWindowHeight * rightDirs[i],
				farCenter + farWindowHeight * upDirs[i] - farWindowHeight * rightDirs[i],
				farCenter - farWindowHeight * upDirs[i] + farWindowHeight * rightDirs[i],
				farCenter - farWindowHeight * upDirs[i] - farWindowHeight * rightDirs[i]};
		Vector3 minPos = frustumPoints[0];
		Vector3 maxPos = frustumPoints[0];
		for (uint j = 1; j < 8; ++j) {
			minPos = Min<Vector3>(minPos, frustumPoints[j]);
			maxPos = Max<Vector3>(maxPos, frustumPoints[j]);
		}
		minMaxBound[i] = {minPos, maxPos};
	};
}
