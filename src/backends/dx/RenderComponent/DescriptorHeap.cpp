#include <RenderComponent/DescriptorHeap.h>
//#endif
#include <RenderComponent/DescriptorHeap.h>
#include <RenderComponent/GPUResourceBase.h>
#include <Singleton/Graphics.h>
#include <PipelineComponent/ThreadCommand.h>
DescriptorHeap::DescriptorHeap(
	GFXDevice* pDevice,
	D3D12_DESCRIPTOR_HEAP_TYPE Type,
	uint64 NumDescriptors,
	bool bShaderVisible) {
	InternalCreate(
		pDevice,
		Type,
		NumDescriptors,
		bShaderVisible);
}
void DescriptorHeap::InternalDispose() {
	if (rootPtr) {
		allocator->Release(handle);
	}
}
void DescriptorHeap::InternalCreate(
	GFXDevice* pDevice,
	D3D12_DESCRIPTOR_HEAP_TYPE Type,
	uint64 NumDescriptors,
	bool bShaderVisible) {
	ElementAllocator* allocator;
	switch (Type) {
		case D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV:
			allocator = Graphics::current->srvAllocator;
			break;
		case D3D12_DESCRIPTOR_HEAP_TYPE_RTV:
			allocator = Graphics::current->rtvAllocator;
			break;
		default:
			allocator = Graphics::current->dsvAllocator;
			break;
	}
	this->allocator = allocator;

	handle = allocator->Allocate(
		NumDescriptors);
	std::cout << handle.node->obj.avaliable;
	rootPtr = handle.GetBlockResource<DescriptorHeapRoot>();
	size = NumDescriptors;
	offset = handle.GetPosition();
}
DescriptorHeap::~DescriptorHeap() {
	InternalDispose();
}
void DescriptorHeap::Create(
	GFXDevice* pDevice,
	D3D12_DESCRIPTOR_HEAP_TYPE Type,
	uint64 NumDescriptors,
	bool bShaderVisible) {
	InternalDispose();
	InternalCreate(
		pDevice,
		Type,
		NumDescriptors,
		bShaderVisible);
}
void DescriptorHeap::SetDescriptorHeap(ThreadCommand* commandList) const {
	commandList->UpdateDescriptorHeap(this, rootPtr);
}
void DescriptorHeap::CreateUAV(GFXDevice* device, GPUResourceBase const* resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* pDesc, uint64 index) {
	rootPtr->CreateUAV(
		device,
		resource,
		pDesc,
		index + offset);
}
void DescriptorHeap::CreateSRV(GFXDevice* device, GPUResourceBase const* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* pDesc, uint64 index) {
	rootPtr->CreateSRV(
		device,
		resource,
		pDesc,
		index + offset);
}
void DescriptorHeap::CreateRTV(GFXDevice* device, GPUResourceBase const* resource, uint depthSlice, uint mipCount, const D3D12_RENDER_TARGET_VIEW_DESC* pDesc, uint64 index) {
	rootPtr->CreateRTV(
		device,
		resource,
		depthSlice,
		mipCount,
		pDesc,
		index + offset);
}
BindType DescriptorHeap::GetBindType(uint64 index) const {
	return rootPtr->GetBindType(index + offset);
}
void DescriptorHeap::CreateDSV(GFXDevice* device, GPUResourceBase const* resource, uint depthSlice, uint mipCount, const D3D12_DEPTH_STENCIL_VIEW_DESC* pDesc, uint64 index) {
	rootPtr->CreateDSV(
		device,
		resource,
		depthSlice,
		mipCount,
		pDesc,
		index + offset);
}
void DescriptorHeap::ClearView(uint64 index) const {
	rootPtr->ClearView(index + offset);
}
