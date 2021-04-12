#include <RenderComponent/DescriptorHeapRoot.h>
//#endif
#include <RenderComponent/DescriptorHeapRoot.h>
#include <RenderComponent/GPUResourceBase.h>
#include <PipelineComponent/ThreadCommand.h>
DescriptorHeapRoot::DescriptorHeapRoot(
	GFXDevice* pDevice,
	D3D12_DESCRIPTOR_HEAP_TYPE Type,
	uint64 NumDescriptors,
	bool bShaderVisible) {
	recorder = new BindData[NumDescriptors];
	memset(recorder, 0, sizeof(BindData) * NumDescriptors);
	Desc.Type = Type;
	Desc.NumDescriptors = NumDescriptors;
	Desc.Flags = (bShaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
	Desc.NodeMask = 0;
	ThrowIfFailed(pDevice->CreateDescriptorHeap(
		&Desc,
		IID_PPV_ARGS(&pDH)));
	hCPUHeapStart = pDH->GetCPUDescriptorHandleForHeapStart();
	hGPUHeapStart = pDH->GetGPUDescriptorHandleForHeapStart();
	HandleIncrementSize = pDevice->GetDescriptorHandleIncrementSize(Desc.Type);
}
DescriptorHeapRoot::~DescriptorHeapRoot() {
	if (recorder) {
		delete recorder;
	}
}
void DescriptorHeapRoot::Create(
	GFXDevice* pDevice,
	D3D12_DESCRIPTOR_HEAP_TYPE Type,
	uint64 NumDescriptors,
	bool bShaderVisible) {
	if (recorder) {
		delete recorder;
	}
	recorder = new BindData[NumDescriptors];
	memset(recorder, 0, sizeof(BindData) * NumDescriptors);
	Desc.Type = Type;
	Desc.NumDescriptors = NumDescriptors;
	Desc.Flags = (bShaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
	Desc.NodeMask = 0;
	ThrowIfFailed(pDevice->CreateDescriptorHeap(
		&Desc,
		IID_PPV_ARGS(&pDH)));
	hCPUHeapStart = pDH->GetCPUDescriptorHandleForHeapStart();
	hGPUHeapStart = pDH->GetGPUDescriptorHandleForHeapStart();
	HandleIncrementSize = pDevice->GetDescriptorHandleIncrementSize(Desc.Type);
}
void DescriptorHeapRoot::CreateUAV(GFXDevice* device, GPUResourceBase const* resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* pDesc, uint64 index) {
	BindData targetBindType = {BindType::UAV, resource->GetInstanceID()};
	{
		std::lock_guard<decltype(mtx)> lck(mtx);
		if (recorder[index] == targetBindType)
			return;
		recorder[index] = targetBindType;
	}
	device->CreateUnorderedAccessView(resource->GetResource(), nullptr, pDesc, hCPU(index));
}
void DescriptorHeapRoot::CreateSRV(GFXDevice* device, GPUResourceBase const* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* pDesc, uint64 index) {
	BindData targetBindType = {BindType::SRV, resource->GetInstanceID()};
	{
		std::lock_guard<decltype(mtx)> lck(mtx);
		if (recorder[index] == targetBindType)
			return;
		recorder[index] = targetBindType;
	}
	device->CreateShaderResourceView(resource->GetResource(), pDesc, hCPU(index));
}
void DescriptorHeapRoot::CreateRTV(GFXDevice* device, GPUResourceBase const* resource, uint depthSlice, uint mipCount, const D3D12_RENDER_TARGET_VIEW_DESC* pDesc, uint64 index) {
	BindData targetBindType = {BindType::RTV, resource->GetInstanceID()};
	{
		std::lock_guard<decltype(mtx)> lck(mtx);
		if (recorder[index] == targetBindType)
			return;
		recorder[index] = targetBindType;
	}
	device->CreateRenderTargetView(resource->GetResource(), pDesc, hCPU(index));
}
BindType DescriptorHeapRoot::GetBindType(uint64 index) const {
	return recorder[index].type;
}
void DescriptorHeapRoot::CreateDSV(GFXDevice* device, GPUResourceBase const* resource, uint depthSlice, uint mipCount, const D3D12_DEPTH_STENCIL_VIEW_DESC* pDesc, uint64 index) {
	BindData targetBindType = {BindType::DSV, resource->GetInstanceID()};
	{
		std::lock_guard<decltype(mtx)> lck(mtx);
		if (recorder[index] == targetBindType)
			return;
		recorder[index] = targetBindType;
	}
	device->CreateDepthStencilView(resource->GetResource(), pDesc, hCPU(index));
}
void DescriptorHeapRoot::ClearView(uint64 index) {
	BindData targetBindType = {BindType::None, (uint64)-1};
	std::lock_guard<decltype(mtx)> lck(mtx);
	recorder[index] = targetBindType;
}
