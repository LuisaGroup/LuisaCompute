#pragma once
#include <Runtime/Device.h>
#include <vstl/ObjectPtr.h>
namespace toolhub::directx {
class DefaultBuffer;
class BottomAccel;
class CommandBufferBuilder;
class ResourceStateTracker;
class Mesh;
class BottomAccel;
class TopAccel : public vstd::IOperatorNewBase {
	friend class BottomAccel;
	vstd::ObjectPtr<DefaultBuffer> instBuffer;
	vstd::ObjectPtr<DefaultBuffer> accelBuffer;
	vstd::HashMap<Buffer const*, void> resourceMap;
	Device* device;
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo;
	D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc;
	mutable std::mutex mtx;
	size_t capacity = 0;
	vstd::vector<BottomAccel const*> accelMap;
	struct CopyCommand {
		vstd::ObjectPtr<DefaultBuffer> srcBuffer;
		vstd::ObjectPtr<DefaultBuffer> dstBuffer;
	};
	struct UpdateCommand {
		D3D12_RAYTRACING_INSTANCE_DESC ist;
		BufferView buffer;
	};
	using Command = vstd::variant<
		CopyCommand,
		UpdateCommand>;
	vstd::vector<Command> delayCommands;
	void UpdateBottomAccel(uint idx, BottomAccel const* c);

public:
	TopAccel(Device* device);
	uint Length() const { return topLevelBuildDesc.Inputs.NumDescs; }
	bool IsBufferInAccel(Buffer const* buffer) const;
	bool IsMeshInAccel(Mesh const* mesh) const;
	bool Update(
		uint idx,
		BottomAccel const* accel,
		uint mask,
		float4x4 const& localToWorld);
	void Emplace(
		BottomAccel const* accel,
		uint mask,
		float4x4 const& localToWorld);
	DefaultBuffer const* GetAccelBuffer() const {
		return accelBuffer ? (DefaultBuffer const*)accelBuffer : (DefaultBuffer const*)nullptr;
	}
	void Reserve(
		size_t newCapacity);
	void Build(
		ResourceStateTracker& tracker,
		CommandBufferBuilder& builder);
	~TopAccel();
};
}// namespace toolhub::directx