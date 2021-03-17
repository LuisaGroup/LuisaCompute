#include "Transform.h"
#include "../WorldManagement/World.h"
#include "../Common/MetaLib.h"
#include "../Common/RandomVector.h"
#include "../JobSystem/JobInclude.h"
#include "../Common/Runnable.h"
#include "../CJsonObject/CJsonObject.hpp"
#include "../CJsonObject/SerializedObject.h"
#include "../LogicComponent/Component.h"
#include "../Common/PoolAllocator.h"
ArrayList<JobHandle> Transform::moveWorldHandles;
using namespace Math;
using namespace neb;
std::mutex Transform::globalMtx;
PoolAllocator<Transform> transform_AllocatePool;
void* Transform::operator new(size_t size) noexcept {
	return transform_AllocatePool.Allocate(size);
}
void Transform::operator delete(void* pdead, size_t size) noexcept {
	transform_AllocatePool.Free(pdead, size);
}
void Transform::Transform_MoveTheWorldLogic(
	std::pair<Transform*, uint*>* startPtr,
	std::pair<Transform*, uint*>* endPtr,
	Math::Vector3 const& moveDirection,
	int3 const& moveBlock) noexcept {
	for (; startPtr != endPtr; startPtr++) {
		auto&& r = startPtr->first;
		if (r->transData.isWorldMovable) {
			{
				std::lock_guard lck(r->localLock);
				r->transData.position = (Vector3)r->transData.position + moveDirection;
				r->transData.worldBlockPos = moveBlock;
			}
			auto&& mtw = r->allMoveTheWorldComponents;
			if (mtw.Length() > 0) {
				for (uint j = 0; j < mtw.Length(); ++j) {
					auto& comp = mtw[j];
					if (comp->enabled)
						comp->OnMovingTheWorld(moveDirection);
				}
			}
		}
	}
}
void Transform::SetRotation(const float4& quaternion) noexcept {
	Matrix4 rotationMatrix = XMMatrixRotationQuaternion(Vector4(quaternion));
	float3 right, up, forward;
	right = normalize((Vector4&)rotationMatrix[0]);
	up = normalize((Vector4&)rotationMatrix[1]);
	forward = normalize((Vector4&)rotationMatrix[2]);
	TransformData& data = transData;
	{
		std::lock_guard lck(localLock);
		data.right = right;
		data.up = up;
		data.forward = forward;
	}
}
struct TransformMoveStruct {
	uint start, end;
	Vector3 moveDirection;
	int3 moveBlock;
	void operator()() noexcept {
		auto&& arr = World::GetInstance()->allTransformsPtr;
		uint len = arr.Length();
		len = Min(len, end);
		if (len <= start) return;
		auto startPtr = arr.GetData() + start;
		auto endPtr = arr.GetData() + len;
		Transform::Transform_MoveTheWorldLogic(startPtr, endPtr, moveDirection, moveBlock);
	}
};
struct TransformMoveToEndStruct {
	uint start;
	Vector3 moveDirection;
	int3 moveBlock;
	void operator()() noexcept {
		auto&& arr = World::GetInstance()->allTransformsPtr;
		uint len = arr.Length();
		if (len <= start) return;
		auto startPtr = arr.GetData() + start;
		auto endPtr = arr.GetData() + len;
		Transform::Transform_MoveTheWorldLogic(startPtr, endPtr, moveDirection, moveBlock);
	}
};
void Transform::MoveTheWorld(int3* worldBlockIndexPtr, const int3& moveBlock, const double3& moveDirection, JobBucket* bucket) noexcept {
	JobHandle lockHandle = bucket->GetTask(nullptr, 0, [=]() -> void {
		globalMtx.lock();
		*worldBlockIndexPtr += moveBlock;
	});
	constexpr uint PER_THREAD_TRANSFORM = 512;
	moveWorldHandles.clear();
	uint jobCount = World::GetInstance()->allTransformsPtr.Length() / PER_THREAD_TRANSFORM;
	uint start = 0;
	uint end = 0;
	TransformMoveStruct moveStruct;
	TransformMoveToEndStruct moveToEndStruct;
	moveStruct.moveDirection = Vector3(moveDirection.x, moveDirection.y, moveDirection.z);
	moveStruct.moveBlock = *worldBlockIndexPtr;
	moveToEndStruct.moveDirection = moveStruct.moveDirection;
	for (uint i = 0; i < jobCount; ++i) {
		start = end;
		end = PER_THREAD_TRANSFORM * (i + 1);
		moveStruct.start = start;
		moveStruct.end = end;
		moveWorldHandles.push_back(bucket->GetTask({lockHandle}, moveStruct));
	}
	moveToEndStruct.start = end;
	moveToEndStruct.moveBlock = *worldBlockIndexPtr;
	moveWorldHandles.push_back(bucket->GetTask({lockHandle}, moveToEndStruct));
	bucket->GetTask(moveWorldHandles.data(), moveWorldHandles.size(), []() -> void {
		globalMtx.unlock();
	});
}
void Transform::CallAfterUpdateEvents() noexcept {
	for (uint i = 0; i < allTransformUpdateComponents.Length(); ++i) {
		auto& comp = allTransformUpdateComponents[i];
		if (comp->enabled)
			comp->OnTransformUpdated();
	}
}
void Transform::SetRight(const float3& right) noexcept {
	{
		std::lock_guard lck(localLock);
		transData.right = right;
	}
	CallAfterUpdateEvents();
}
void Transform::SetUp(const float3& up) noexcept {
	{
		std::lock_guard lck(localLock);
		transData.up = up;
	}
	CallAfterUpdateEvents();
}
void Transform::SetForward(const float3& forward) noexcept {
	{
		std::lock_guard lck(localLock);
		transData.forward = forward;
	}
	CallAfterUpdateEvents();
}
void Transform::SetPosition(const float3& position) noexcept {
	{
		std::lock_guard lck(localLock);
		transData.position = position;
	}
	CallAfterUpdateEvents();
}
double3 Transform::GetAbsolutePosition() const noexcept {
	float3 localPos;
	int3 worldBlockOffset;
	{
		std::lock_guard lck(localLock);
		localPos = transData.position;
		worldBlockOffset = transData.worldBlockPos;
	}
	double3 worldBlockCoord = double3(
		(double)worldBlockOffset.x * (double)World::BLOCK_SIZE + (double)localPos.x,
		(double)worldBlockOffset.y * (double)World::BLOCK_SIZE + (double)localPos.y,
		(double)worldBlockOffset.z * (double)World::BLOCK_SIZE + (double)localPos.z);
	return worldBlockCoord;
}
Transform::Transform(SerializedObject* jsonObj, bool isWorldMovable) noexcept {
	World* world = World::GetInstance();
	if (world != nullptr) {
		lockGuard lck(globalMtx);
		world->allTransformsPtr.Add(this, (uint*)&worldIndex);
		if (isWorldMovable)
			transData.worldBlockPos = World::GetInstance()->blockIndex;
		else
			transData.worldBlockPos = {0, 0, 0};
	} else {
		worldIndex = -1;
	}
	vengine::string s;
	double3 pos = {0, 0, 0};
	float4 rot = {0, 0, 0, 1};
	float3 scale = {1, 1, 1};
	if (jsonObj->Get("position", s)) {
		ReadStringToDoubleVector<double3>(s.data(), s.length(), &pos);
	}
	if (jsonObj->Get("rotation", s)) {
		ReadStringToVector<float4>(s.data(), s.length(), &rot);
	}
	if (jsonObj->Get("localscale", s)) {
		ReadStringToVector<float3>(s.data(), s.length(), &scale);
	}
	Vector4 quaternion = rot;
	Matrix4 rotationMatrix = XMMatrixRotationQuaternion(quaternion);
	float3 right, up, forward;
	right = normalize((Vector3 const&)rotationMatrix[0]);
	up = normalize((Vector3 const&)rotationMatrix[1]);
	forward = normalize((Vector3 const&)rotationMatrix[2]);
	std::lock_guard lck(localLock);
	pos -= double3(World::BLOCK_SIZE) * double3(transData.worldBlockPos.x, transData.worldBlockPos.y, transData.worldBlockPos.z);
	transData = {
		up,
		forward,
		right,
		scale,
		{(float)pos.x, (float)pos.y, (float)pos.z},
		isWorldMovable};
}
Transform::Transform(
	const double3& pos,
	const Vector4& rot,
	const Vector3& scale,
	bool isWorldMovable) noexcept {
	World* world = World::GetInstance();
	if (world != nullptr) {
		lockGuard lck(globalMtx);
		world->allTransformsPtr.Add(this, (uint*)&worldIndex);
		if (isWorldMovable)
			transData.worldBlockPos = World::GetInstance()->blockIndex;
		else
			transData.worldBlockPos = {0, 0, 0};
	} else {
		worldIndex = -1;
	}
	Vector4 quaternion = rot;
	Matrix4 rotationMatrix = XMMatrixRotationQuaternion(quaternion);
	float3 right, up, forward;
	right = normalize((Vector3 const&)rotationMatrix[0]);
	up = normalize((Vector3 const&)rotationMatrix[1]);
	forward = normalize((Vector3 const&)rotationMatrix[2]);
	std::lock_guard lck(localLock);
	double3 newPos = pos - double3(World::BLOCK_SIZE) * double3(transData.worldBlockPos.x, transData.worldBlockPos.y, transData.worldBlockPos.z);
	transData = {
		up,
		forward,
		right,
		scale,
		float3(newPos.x, newPos.y, newPos.z),
		isWorldMovable};
}
Transform::Transform(
	const double3& pos,
	const Math::Vector3& right, 
	const Math::Vector3& up, 
	const Math::Vector3& forward,
	const Math::Vector3& localScale,
	bool isWorldMovable) noexcept {
	World* world = World::GetInstance();
	if (world != nullptr) {
		lockGuard lck(globalMtx);
		world->allTransformsPtr.Add(this, (uint*)&worldIndex);
		if (isWorldMovable)
			transData.worldBlockPos = World::GetInstance()->blockIndex;
		else
			transData.worldBlockPos = {0, 0, 0};
	} else {
		worldIndex = -1;
	}
	std::lock_guard lck(localLock);
	double3 newPos = pos - double3(World::BLOCK_SIZE) * double3(transData.worldBlockPos.x, transData.worldBlockPos.y, transData.worldBlockPos.z);
	transData = {
		up,
		forward,
		right,
		localScale,
		float3(newPos.x, newPos.y, newPos.z),
		isWorldMovable};
}
Transform::Transform(
	bool isWorldMovable) noexcept {
	World* world = World::GetInstance();
	if (world != nullptr) {
		lockGuard lck(globalMtx);
		world->allTransformsPtr.Add(this, (uint*)&worldIndex);
		if (isWorldMovable)
			transData.worldBlockPos = World::GetInstance()->blockIndex;
		else
			transData.worldBlockPos = {0, 0, 0};
	} else {
		worldIndex = -1;
	}
	std::lock_guard lck(localLock);
	transData = {
		float3(0, 1, 0),
		float3(0, 0, 1),
		float3(1, 0, 0),
		float3(1, 1, 1),
		float3(0, 0, 0),
		isWorldMovable};
}
void Transform::SetLocalScale(const float3& localScale) noexcept {
	{
		std::lock_guard lck(localLock);
		transData.localScale = localScale;
	}
	CallAfterUpdateEvents();
}
void Transform::SetAll(
	const float3& right,
	const float3& up,
	const float3& forward,
	const float3& position,
	const float3& localScale) noexcept {
	auto&& a = transData;
	{
		std::lock_guard lck(localLock);
		a.forward = forward;
		a.right = right;
		a.up = up;
		a.position = position;
		a.localScale = localScale;
	}
	CallAfterUpdateEvents();
}
void Transform::SetAll(
	const float3& position,
	const float4& quaternion,
	const float3& localScale) noexcept {
	{
		Matrix4 rotationMatrix = XMMatrixRotationQuaternion(Vector4(quaternion));
		float3 right, up, forward;
		right = normalize((Vector3&)rotationMatrix[0]);
		up = normalize((Vector3&)rotationMatrix[1]);
		forward = normalize((Vector3&)rotationMatrix[2]);
		auto&& a = transData;
		{
			std::lock_guard lck(localLock);
			a.forward = forward;
			a.right = right;
			a.up = up;
			a.position = position;
			a.localScale = localScale;
		}
	}
	CallAfterUpdateEvents();
}
Matrix4 Transform::GetLocalToWorldMatrixCPU() const noexcept {
	Matrix4 target;
	{
		TransformData const& data = transData;
		Vector3 vec = data.right;
		vec *= data.localScale.x;
		target[0] = vec;
		vec = data.up;
		vec *= data.localScale.y;
		target[1] = vec;
		vec = data.forward;
		vec *= data.localScale.z;
		target[2] = vec;
		/*target[0].m128_f32[3] = data.position.x;
		target[1].m128_f32[3] = data.position.y;
		target[2].m128_f32[3] = data.position.z;*/
		target[3] = Vector4(data.position, 1);
	}
	return transpose(target);
}
Matrix4 Transform::GetLocalToWorldMatrixGPU() const noexcept {
	return GetLocalToWorldMatrixCPU();
}
Matrix4 Transform::GetWorldToLocalMatrixCPU() const noexcept {
	return inverse(GetLocalToWorldMatrixCPU());
}
Matrix4 Transform::GetWorldToLocalMatrixGPU() const noexcept {
	return inverse(GetLocalToWorldMatrixGPU());
}
Component* Transform::GetComponent(Type type) noexcept {
	for (uint i = 0; i < allComponents.Length(); ++i) {
		if (allComponents[i]->GetType() == type) {
			return allComponents[i];
		}
	}
	return nullptr;
}
Transform::~Transform() noexcept {
	for (uint i = 0; i < allComponents.Length(); ++i) {
		if (allComponents[i]) {
			delete allComponents[i];
		}
	}
	World* world = World::GetInstance();
	if (world) {
		if (worldIndex >= 0) {
			lockGuard lck(globalMtx);
			world->allTransformsPtr.Remove(worldIndex);
		}
	}
}
