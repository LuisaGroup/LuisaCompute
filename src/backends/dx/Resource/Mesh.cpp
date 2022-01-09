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
	meshInstance = device->AllocateMeshBuffer();
	vboIdx = device->globalHeap->AllocateIndex();
	iboIdx = device->globalHeap->AllocateIndex();
	meshInstIdx = device->globalHeap->AllocateIndex();
	auto vSrvDesc = vHandle->GetColorSrvDesc(vOffset, vStride * vCount);
	auto iSrvDesc = iHandle->GetColorSrvDesc(iOffset, sizeof(uint) * iCount);
	auto instIdx = meshInstance.buffer->GetColorSrvDesc(meshInstance.offset, meshInstance.byteSize, false);
	if (!vSrvDesc || !iSrvDesc || !instIdx) {
		VEngine_Log("illegal mesh buffer!\n");
		VENGINE_EXIT;
	}
	device->globalHeap->CreateSRV(
		meshInstance.buffer->GetResource(), *instIdx, meshInstIdx);
	device->globalHeap->CreateSRV(
		vHandle->GetResource(), *vSrvDesc, vboIdx);
	device->globalHeap->CreateSRV(
		iHandle->GetResource(), *iSrvDesc, iboIdx);
}
Mesh::~Mesh() {
	device->globalHeap->ReturnIndex(vboIdx);
	device->globalHeap->ReturnIndex(iboIdx);
	device->globalHeap->ReturnIndex(meshInstIdx);
	device->DeAllocateMeshBuffer(meshInstance);
	//TODO: deallocate buffer
}
void Mesh::Build(
	ResourceStateTracker& tracker,
	CommandBufferBuilder& cmd) {
	tracker.RecordState(
		meshInstance.buffer,
		D3D12_RESOURCE_STATE_COPY_SOURCE);
	auto disp = vstd::create_disposer([&] {
		tracker.RecordState(meshInstance.buffer);
	});
	tracker.UpdateState(cmd);
	cmd.Upload(
		meshInstance,
		vstd::get_rvalue_ptr(MeshInstance{
			vboIdx,
			iboIdx,
			uint(vStride),
			uint(vCount),
			uint(iCount)}));
}
ID3D12Resource* Mesh::GetResource() const {
	return meshInstance.buffer->GetResource();
}
D3D12_RESOURCE_STATES Mesh::GetInitState() const {
	return meshInstance.buffer->GetInitState();
}
}// namespace toolhub::directx