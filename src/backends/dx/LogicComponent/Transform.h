#pragma once
#include <VEngineConfig.h>
#include <Common/VObject.h>
#include <LogicComponent/Component.h>
#include <Common/RandomVector.h>
#include <Common/MetaLib.h>
class World;
class AssetReference;
class JobHandle;
class JobBucket;
class Transform;
class TransformMoveStruct;
class TransformMoveToEndStruct;

namespace neb {
class CJsonObject;
}
class SerializedObject;
struct TransformData {
	float3 up;
	float3 forward;
	float3 right;
	float3 localScale;
	float3 position;
	bool isWorldMovable = false;
	int3 worldBlockPos;
};
class Transform final : public VObject {
	friend class Scene;
	friend class Component;
	friend class AssetReference;
	friend class TransformMoveStruct;
	friend class TransformMoveToEndStruct;
	friend class World;

private:
	static std::mutex globalMtx;
	static ArrayList<JobHandle> moveWorldHandles;
	TransformData transData;
	mutable spin_mutex localLock;
	mutable spin_mutex compLock;
	RandomVector<Component*, true> allComponents;
	RandomVector<Component*, true> allTransformUpdateComponents;
	RandomVector<Component*, true> allMoveTheWorldComponents;
	int32_t worldIndex;
	Component* GetComponent(Type type) noexcept;
	static void Transform_MoveTheWorldLogic(
		std::pair<Transform*, uint*>* startPtr,
		std::pair<Transform*, uint*>* endPtr,
		Math::Vector3 const& moveDirection,
		int3 const& moveBlock) noexcept;
	void CallAfterUpdateEvents() noexcept;
	static void MoveTheWorld(int3* worldBlockIndexPtr, const int3& moveBlock, const double3& moveDirection, JobBucket* bucket) noexcept;

public:
	static void* operator new(size_t size) noexcept;
	static void operator delete(void* pdead, size_t size) noexcept;
	vengine::string name;
	~Transform() noexcept;

	template<typename T>
	T* GetComponent() noexcept {
		return (T*)GetComponent(typeid(T));
	}
	uint GetComponentCount() const noexcept {
		return allComponents.Length();
	}
	Transform(SerializedObject* path, bool isWorldMovable = true) noexcept;
	Transform(bool isWorldMovable = true) noexcept;
	Transform(const double3& position, const Math::Vector4& rotation, const Math::Vector3& localScale, bool isWorldMovable = true) noexcept;
	Transform(const double3& position, const Math::Vector3& right, const Math::Vector3& up, const Math::Vector3& forward, const Math::Vector3& localScale, bool isWorldMovable = true) noexcept;
	double3 GetAbsolutePosition() const noexcept;
	float3 GetPosition() const noexcept {
		return transData.position;
	}
	float3 GetForward() const noexcept {
		return transData.forward;
	}
	float3 GetRight() const noexcept {
		return transData.right;
	}
	float3 GetUp() const noexcept {
		return transData.up;
	}
	float3 GetLocalScale() const noexcept {
		return transData.localScale;
	}
	void SetRight(const float3& right) noexcept;
	void SetUp(const float3& up) noexcept;
	void SetForward(const float3& forward) noexcept;
	void SetRotation(const float4& quaternion) noexcept;
	void SetPosition(const float3& position) noexcept;
	void SetLocalScale(const float3& localScale) noexcept;
	void SetAll(
		const float3& right,
		const float3& up,
		const float3& forward,
		const float3& position,
		const float3& localScale) noexcept;
	void SetAll(
		const float3& position,
		const float4& quaternion,
		const float3& localScale) noexcept;
	Math::Matrix4 GetLocalToWorldMatrixCPU() const noexcept;
	Math::Matrix4 GetLocalToWorldMatrixGPU() const noexcept;
	Math::Matrix4 GetWorldToLocalMatrixCPU() const noexcept;
	Math::Matrix4 GetWorldToLocalMatrixGPU() const noexcept;
	KILL_COPY_CONSTRUCT(Transform)
};
