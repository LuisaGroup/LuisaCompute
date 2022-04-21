#pragma once
#include <d3dx12.h>
#include <Resource/Resource.h>
#include <Resource/BufferView.h>
namespace toolhub::directx {
class CommandBufferBuilder;
class ResourceStateTracker;
class Mesh final : public Resource {

public:
	Buffer const* vHandle;
	Buffer const* iHandle;
	uint vOffset;
	uint iOffset;
	uint vStride;
	uint vCount;
	uint iCount;
	Tag GetTag() const { return Tag::Mesh; }
	Mesh(Device* device,
		 Buffer const* vHandle, size_t vOffset, size_t vStride, size_t vCount,
		 Buffer const* iHandle, size_t iOffset, size_t iCount);
	~Mesh();
	void Build(
		ResourceStateTracker& tracker) const;
	VSTD_SELF_PTR
	KILL_MOVE_CONSTRUCT(Mesh)
	KILL_COPY_CONSTRUCT(Mesh)
};
}// namespace toolhub::directx