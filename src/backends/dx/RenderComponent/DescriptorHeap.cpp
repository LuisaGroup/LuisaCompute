#include <RenderComponent/DescriptorHeap.h>
#include <RenderComponent/GPUResourceBase.h>
#include <PipelineComponent/ThreadCommand.h>
DescriptorHeap::DescriptorHeap(
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
	ThrowIfFailed(pDevice->device()->CreateDescriptorHeap(
		&Desc,
		IID_PPV_ARGS(&pDH)));
	hCPUHeapStart = pDH->GetCPUDescriptorHandleForHeapStart();
	hGPUHeapStart = pDH->GetGPUDescriptorHandleForHeapStart();
	HandleIncrementSize = pDevice->device()->GetDescriptorHandleIncrementSize(Desc.Type);
}
DescriptorHeap::~DescriptorHeap() {
	if (recorder) {
		delete recorder;
	}
}
void DescriptorHeap::Create(
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
	ThrowIfFailed(pDevice->device()->CreateDescriptorHeap(
		&Desc,
		IID_PPV_ARGS(&pDH)));
	hCPUHeapStart = pDH->GetCPUDescriptorHandleForHeapStart();
	hGPUHeapStart = pDH->GetGPUDescriptorHandleForHeapStart();
	HandleIncrementSize = pDevice->device()->GetDescriptorHandleIncrementSize(Desc.Type);
}
void DescriptorHeap::CreateUAV(GFXDevice* device, GPUResourceBase const* resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* pDesc, uint64 index) {
	BindData targetBindType = {BindType::UAV, resource->GetInstanceID()};
	{
		std::lock_guard<decltype(mtx)> lck(mtx);
		if (recorder[index] == targetBindType)
			return;
		recorder[index] = targetBindType;
	}
	device->device()->CreateUnorderedAccessView(resource->GetResource(), nullptr, pDesc, hCPU(index));
}
void DescriptorHeap::SetDescriptorHeap(ThreadCommand* tCmd) const {
	tCmd->UpdateDescriptorHeap(this);
}
void DescriptorHeap::CreateSRV(GFXDevice* device, GPUResourceBase const* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* pDesc, uint64 index) {
	BindData targetBindType = {BindType::SRV, resource->GetInstanceID()};
	{
		std::lock_guard<decltype(mtx)> lck(mtx);
		if (recorder[index] == targetBindType)
			return;
		recorder[index] = targetBindType;
	}
	device->device()->CreateShaderResourceView(resource->GetResource(), pDesc, hCPU(index));
}
void DescriptorHeap::CreateRTV(GFXDevice* device, GPUResourceBase const* resource, uint depthSlice, uint mipCount, const D3D12_RENDER_TARGET_VIEW_DESC* pDesc, uint64 index) {
	BindData targetBindType = {BindType::RTV, resource->GetInstanceID()};
	{
		std::lock_guard<decltype(mtx)> lck(mtx);
		if (recorder[index] == targetBindType)
			return;
		recorder[index] = targetBindType;
	}
	device->device()->CreateRenderTargetView(resource->GetResource(), pDesc, hCPU(index));
}
BindType DescriptorHeap::GetBindType(uint64 index) const {
	return recorder[index].type;
}
void DescriptorHeap::CreateDSV(GFXDevice* device, GPUResourceBase const* resource, uint depthSlice, uint mipCount, const D3D12_DEPTH_STENCIL_VIEW_DESC* pDesc, uint64 index) {
	BindData targetBindType = {BindType::DSV, resource->GetInstanceID()};
	{
		std::lock_guard<decltype(mtx)> lck(mtx);
		if (recorder[index] == targetBindType)
			return;
		recorder[index] = targetBindType;
	}
	device->device()->CreateDepthStencilView(resource->GetResource(), pDesc, hCPU(index));
}
void DescriptorHeap::ClearView(uint64 index) {
	BindData targetBindType = {BindType::None, (uint64)-1};
	std::lock_guard<decltype(mtx)> lck(mtx);
	recorder[index] = targetBindType;
}

void DescriptorHeap::CreateSampler(
	GFXDevice* device,
	VObject* resource,
	D3D12_SAMPLER_DESC const* sampDesc,
	uint64 index) {
	BindData targetBindType = {BindType::Sampler, resource->GetInstanceID()};
	{
		std::lock_guard<decltype(mtx)> lck(mtx);
		if (recorder[index] == targetBindType)
			return;
		recorder[index] = targetBindType;
	}
	device->device()->CreateSampler(
		sampDesc, hCPU(index));
}