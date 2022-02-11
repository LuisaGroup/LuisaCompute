#pragma vengine_package vengine_directx
#include <Resource/Mesh.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <Resource/MeshInstance.h>
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
	//TODO: allocate buffer
	vboIdx = device->globalHeap->AllocateIndex();
	iboIdx = device->globalHeap->AllocateIndex();
	auto vSrvDesc = vHandle->GetColorSrvDesc(vOffset, vStride * vCount);
	auto iSrvDesc = iHandle->GetColorSrvDesc(iOffset, sizeof(uint) * iCount);
	if (!vSrvDesc || !iSrvDesc) {
		VEngine_Log("illegal mesh buffer!\n");
		VENGINE_EXIT;
	}

	device->globalHeap->CreateSRV(
		vHandle->GetResource(), *vSrvDesc, vboIdx);
	device->globalHeap->CreateSRV(
		iHandle->GetResource(), *iSrvDesc, iboIdx);
}
Mesh::~Mesh() {
	device->globalHeap->ReturnIndex(vboIdx);
	device->globalHeap->ReturnIndex(iboIdx);
	//TODO: deallocate buffer
}
void Mesh::Build(
	ResourceStateTracker& tracker) const {
    tracker.RecordState(
        vHandle,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    tracker.RecordState(
        iHandle,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
	
}
}// namespace toolhub::directx