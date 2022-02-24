#pragma vengine_package vengine_directx
#include <Resource/Mesh.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <DXRuntime/CommandBuffer.h>
#include <DXRuntime/CommandAllocator.h>
#include <DXRuntime/ResourceStateTracker.h>
namespace toolhub::directx {
Mesh::Mesh(Device* device,
		   Buffer const* vHandle, size_t vOffset, size_t vStride, size_t vCount,
		   Buffer const* iHandle, size_t iOffset, size_t iCount)
	: Resource(device),
	  vHandle(vHandle),
	  iHandle(iHandle),
	  vOffset(vOffset),
	  iOffset(iOffset),
	  vStride(vStride),
	  vCount(vCount),
	  iCount(iCount) {
}
Mesh::~Mesh() {
}
void Mesh::Build(
	ResourceStateTracker& tracker) const {
    tracker.RecordState(
        vHandle,
        VEngineShaderResourceState);
    tracker.RecordState(
        iHandle,
        VEngineShaderResourceState);
	
}
}// namespace toolhub::directx