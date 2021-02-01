#pragma once
#include "../Common/Common.h"
#include "../Common/DLL.h"

#define Deg2Rad 0.0174532924
#define Rad2Deg  57.29578
#define Float32MaxValue  3.40282347E+37f
#define Float32MinValue  -3.40282347E+37f
#define Float32Episilon  1.401298E-45f
#define Float64MaxValue  1.7976931348623157E+307
#define Float64MinValue  -1.7976931348623157E+307
#define Float64Episilon  4.94065645841247E-324

struct Cone
{
	float3 vertex;
	float height;
	float3 direction;
	float radius;
	Cone(const float3& position, float distance, const float3& direction, float angle) noexcept :
		vertex(position),
		height(distance),
		direction(direction)
	{
		radius = tan(angle * 0.5 * Deg2Rad) * height;
	}
};
class  MathLib final
{
public:
	MathLib() = delete;
	~MathLib() = delete;
	static Math::Vector4 GetPlane(
		const Math::Vector3& normal,
		const Math::Vector3& inPoint) noexcept;
	static Math::Vector4 GetPlane(
		const Math::Vector3& a,
		const Math::Vector3& b,
		const Math::Vector3& c) noexcept;
	/*static bool BoxIntersect(
		const Math::Matrix4& localToWorldMatrix,
		Math::Vector4* planes,
		const Math::Vector3& localPosition,
		const Math::Vector3& localExtent);*/
	static bool BoxIntersect(
		const Math::Matrix4& localToWorldMatrix,
		Math::Vector4* planes,
		const Math::Vector3& localPosition,
		const Math::Vector3& localExtent,
		const Math::Vector3& frustumMinPoint,
		const Math::Vector3& frustumMaxPoint) noexcept;
	/*static void GetCameraNearPlanePoints(
		const Math::Matrix4& localToWorldMatrix,
		double fov,
		double aspect,
		double distance,
		Math::Vector3* corners
	);*/

	static void GetCameraNearPlanePoints(
		const Math::Vector3& right,
		const Math::Vector3& up,
		const Math::Vector3& forward,
		const Math::Vector3& position,
		double fov,
		double aspect,
		double distance,
		Math::Vector3* corners
	) noexcept;
	/*
	static void GetPerspFrustumPlanes(
		const Math::Matrix4& localToWorldMatrix,
		double fov,
		double aspect,
		double nearPlane,
		double farPlane,
		float4* frustumPlanes);*/
	static void GetPerspFrustumPlanes(
		const Math::Vector3& right,
		const Math::Vector3& up,
		const Math::Vector3& forward,
		const Math::Vector3& position,
		double fov,
		double aspect,
		double nearPlane,
		double farPlane,
		Math::Vector4* frustumPlanes) noexcept;
	static float GetScreenMultiple(
		float fov,
		float aspect
	) noexcept;
	static float GetScreenPercentage(
		Math::Vector3 const& boundsOrigin,
		Math::Vector3 const& camOrigin,
		float screenMultiple,
		float sphereRadius
	) noexcept;
	/*static void GetPerspFrustumPlanes(
		const Math::Matrix4& localToWorldMatrix,
		double fov,
		double aspect,
		double nearPlane,
		double farPlane,
		Math::Vector4* frustumPlanes
	);

	static void GetFrustumBoundingBox(
		const Math::Matrix4& localToWorldMatrix,
		double nearWindowHeight,
		double farWindowHeight,
		double aspect,
		double nearZ,
		double farZ,
		Math::Vector3* minValue,
		Math::Vector3* maxValue
	);*/
	static void GetFrustumBoundingBox(
		const Math::Vector3& right,
		const Math::Vector3& up,
		const Math::Vector3& forward,
		const Math::Vector3& position,
		double nearWindowHeight,
		double farWindowHeight,
		double aspect,
		double nearZ,
		double farZ,
		Math::Vector3* minValue,
		Math::Vector3* maxValue
	) noexcept;

	static Math::Matrix4 GetOrthoMatrix(
		float nearPlane,
		float farPlane,
		float size,
		float aspect,
		bool renderTarget,
		bool gpuResource) noexcept;

	static float GetPointDistanceToPlane(
		const Math::Vector4& plane,
		const Math::Vector3& point) noexcept
	{
		float dotValue = dot((Math::Vector3)plane, (Math::Vector3)point);
		return dotValue + plane.GetW();
	}
	static bool ConeIntersect(const Cone& cone, const Math::Vector4& plane) noexcept;
	static void GetOrthoCamFrustumPlanes(
		const Math::Vector3& right,
		const Math::Vector3& up,
		const Math::Vector3& forward,
		const Math::Vector3& position,
		float xSize,
		float ySize,
		float nearPlane,
		float farPlane,
		Math::Vector4* results) noexcept;
	static void GetOrthoCamFrustumPoints(
		const Math::Vector3& right,
		const Math::Vector3& up,
		const Math::Vector3& forward,
		const Math::Vector3& position,
		float xSize,
		float ySize,
		float nearPlane,
		float farPlane,
		Math::Vector3* results) noexcept;

	static float DistanceToCube(const Math::Vector3& cubeSize, const Math::Vector3& relativePosToCube) noexcept;
	static double DistanceToQuad(double size, float2 quadToTarget) noexcept;
	static bool BoxIntersect(const Math::Vector3& position, const Math::Vector3& extent, Math::Vector4* planes) noexcept;
	static bool BoxIntersect(const Math::Vector3& position,
		const Math::Vector3& extent,
		Math::Vector4* planes,
		const Math::Vector3& frustumMinPoint, 
		const Math::Vector3& frustumMaxPoint) noexcept;
	static bool BoxContactWithBox(const double3& min0, const double3& max0, const double3& min1, const double3& max1) noexcept;
	static Math::Vector4 CameraSpacePlane(const Math::Matrix4& worldToCameraMatrix, const Math::Vector3& pos, const Math::Vector3& normal, float clipPlaneOffset) noexcept;
	static Math::Matrix4 CalculateObliqueMatrix(const Math::Vector4& clipPlane, const Math::Matrix4& projection) noexcept;
	static void GetCubemapRenderData(
		const Math::Vector3& position,
		float nearZ,
		float farZ,
		Math::Matrix4* viewProj,
		Math::Vector4** frustumPlanes,
		std::pair<Math::Vector3, Math::Vector3>* minMaxBound) noexcept;
	static float CalculateTextureMipQFactor(
		float fov,
		float distance,
		uint camRes,
		float texRes,
		float panelSize
	) noexcept;
	static float CalculateMipDistance(
		float fov,
		float mip,
		uint camRes,
		float texRes,
		float panelSize) noexcept;
	static float GetScreenPercentage(
		Math::Matrix4 const& projectMat,
		float objectToCameraDistance,
		float sphereRadius
	);
	static double GetHalton_Float(uint index);
	static double2 GetHalton_Vector2(uint index);
	static double3 GetHalton_Vector3(uint index);
};