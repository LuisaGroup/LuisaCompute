#pragma vengine_package vengine_directx
#include <Resource/DescriptorHeap.h>
namespace toolhub::directx {
DescriptorHeap::DescriptorHeap(
    Device *device,
    D3D12_DESCRIPTOR_HEAP_TYPE Type,
    uint64 numDescriptors,
    bool bShaderVisible)
    : Resource(device),
      allocatePool(numDescriptors),
      numDescriptors(numDescriptors) {
    Desc.Type = Type;
    Desc.NumDescriptors = numDescriptors;
    Desc.Flags = (bShaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
    Desc.NodeMask = 0;
    {
        for (auto i : vstd::range(numDescriptors)) {
            allocatePool[i] = i;
        }
    }
    ThrowIfFailed(device->device->CreateDescriptorHeap(
        &Desc,
        IID_PPV_ARGS(&pDH)));
    hCPUHeapStart = pDH->GetCPUDescriptorHandleForHeapStart();
    hGPUHeapStart = pDH->GetGPUDescriptorHandleForHeapStart();
    HandleIncrementSize = device->device->GetDescriptorHandleIncrementSize(Desc.Type);
}
DescriptorHeap::~DescriptorHeap() {
}
uint DescriptorHeap::AllocateIndex() {
    std::lock_guard lck(heapMtx);
#ifdef DEBUG
    if (allocatePool.empty()) {
        VEngine_Log("bindless allocator out or range!\n");
        VENGINE_EXIT;
    }
#endif
    uint v = allocatePool.erase_last();
    return v;
}
void DescriptorHeap::ReturnIndex(uint v) {
    std::lock_guard lck(heapMtx);
    allocatePool.emplace_back(v);
}
void DescriptorHeap::Reset() {
    allocatePool.resize(numDescriptors);

    for (auto i : vstd::range(numDescriptors)) {
        allocatePool[i] = i;
    }
}
void DescriptorHeap::CreateUAV(ID3D12Resource *resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC &pDesc, uint64 index) {
    device->device->CreateUnorderedAccessView(resource, nullptr, &pDesc, hCPU(index));
}
void DescriptorHeap::CreateSRV(ID3D12Resource *resource, const D3D12_SHADER_RESOURCE_VIEW_DESC &pDesc, uint64 index) {
    device->device->CreateShaderResourceView(resource, &pDesc, hCPU(index));
}
void DescriptorHeap::CreateSampler(D3D12_SAMPLER_DESC const &desc, uint64 index) {
    device->device->CreateSampler(&desc, hCPU(index));
}
}// namespace toolhub::directx