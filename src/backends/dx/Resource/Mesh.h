#pragma once
#include <d3dx12.h>
#include <Resource/Resource.h>
#include <Resource/BufferView.h>
namespace toolhub::directx {
class CommandBufferBuilder;
class ResourceStateTracker;
class Mesh final : public Resource {
	BufferView meshInstance;
	uint vboIdx = std::numeric_limits<uint>::max();
	uint iboIdx = std::numeric_limits<uint>::max();

public:
	Buffer const* vHandle;
	Buffer const* iHandle;
	uint vOffset;
	uint iOffset;
	uint vStride;
	uint vCount;
	uint iCount;
	uint meshInstIdx;
	BufferView GetMeshInstance() const { return meshInstance; }
	uint GetMeshInstIdx() const { return meshInstIdx; }
	Tag GetTag() const { return Tag::Mesh; }
	Mesh(Device* device,
		 Buffer const* vHandle, size_t vOffset, size_t vStride, size_t vCount,
		 Buffer const* iHandle, size_t iOffset, size_t iCount);
	~Mesh();
	void Build(
		ResourceStateTracker& tracker,
		CommandBufferBuilder& cmd) const;
	ID3D12Resource* GetResource() const override;
	D3D12_RESOURCE_STATES GetInitState() const override;
	VSTD_SELF_PTR
	KILL_MOVE_CONSTRUCT(Mesh)
	KILL_COPY_CONSTRUCT(Mesh)
};
}// namespace toolhub::directx