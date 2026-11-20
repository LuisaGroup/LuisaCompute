#include <Resource/DescriptorHeap.h>
namespace lc::dx {
DescriptorHeap::DescriptorHeap(
    Device *device,
    D3D12_DESCRIPTOR_HEAP_TYPE Type,
    uint32_t numDescriptors,
    bool bShaderVisible)
    : Resource(device),
      numDescriptors(numDescriptors),
      subAlloc(numDescriptors, 1) {
    allocIndex = 0;
    Desc.Type = Type;
    Desc.NumDescriptors = numDescriptors;
    Desc.Flags = (bShaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
    Desc.NodeMask = 0;
    ThrowIfFailed(device->device->CreateDescriptorHeap(
        &Desc,
        IID_PPV_ARGS(&pDH)));
    hCPUHeapStart = pDH->GetCPUDescriptorHandleForHeapStart();
    if (bShaderVisible) {
        hGPUHeapStart = pDH->GetGPUDescriptorHandleForHeapStart();
    } else {
        hGPUHeapStart.ptr = 0;
    }
    HandleIncrementSize = device->device->GetDescriptorHandleIncrementSize(Desc.Type);
}
DescriptorHeap::~DescriptorHeap() {
}
D3D12_GPU_DESCRIPTOR_HANDLE DescriptorHeap::hGPU(uint64 index) const {
    D3D12_GPU_DESCRIPTOR_HANDLE h = {hGPUHeapStart.ptr + index * HandleIncrementSize};
    return h;
}
D3D12_CPU_DESCRIPTOR_HANDLE DescriptorHeap::hCPU(uint64 index) const {
    D3D12_CPU_DESCRIPTOR_HANDLE h = {hCPUHeapStart.ptr + index * HandleIncrementSize};
    return h;
}
uint32_t DescriptorHeap::AllocateIndex() {
    std::lock_guard lck(heapMtx);
    if (freeList.empty()) {
        if (allocIndex == numDescriptors) [[unlikely]] {
            vengine_log("bindless allocator out or range!\n");
            VENGINE_EXIT;
        }
        auto v = allocIndex;
        allocIndex++;
        return v;
    }
    auto i = freeList.back();
    freeList.pop_back();
    return i;
}
void DescriptorHeap::ReturnIndex(uint32_t v) {
    std::lock_guard lck(heapMtx);
    freeList.emplace_back(v);
}
void DescriptorHeap::Reset() {
    freeList.clear();
    allocIndex = 0;
}
void DescriptorHeap::CreateUAV(ID3D12Resource *resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC &pDesc, uint64 index) {
    device->device->CreateUnorderedAccessView(resource, nullptr, &pDesc, hCPU(index));
}
void DescriptorHeap::CreateSRV(ID3D12Resource *resource, const D3D12_SHADER_RESOURCE_VIEW_DESC &pDesc, uint64 index) {
    device->device->CreateShaderResourceView(resource, &pDesc, hCPU(index));
}
void DescriptorHeap::CreateRTV(ID3D12Resource *resource, const D3D12_RENDER_TARGET_VIEW_DESC &pDesc, uint64 index) {
    device->device->CreateRenderTargetView(resource, &pDesc, hCPU(index));
}
void DescriptorHeap::CreateDSV(ID3D12Resource *resource, const D3D12_DEPTH_STENCIL_VIEW_DESC &pDesc, uint64 index) {
    device->device->CreateDepthStencilView(resource, &pDesc, hCPU(index));
}
void DescriptorHeap::CreateSampler(D3D12_SAMPLER_DESC const &desc, uint64 index) {
    device->device->CreateSampler(&desc, hCPU(index));
}
luisa::FirstFit::Node *DescriptorHeap::SubAllocate(uint32_t size) {
    std::lock_guard lck(heapMtx);
    auto ptr = subAlloc.allocate_best_fit(size);
    if (ptr == nullptr || ptr->offset() + ptr->size() > (numDescriptors - allocIndex)) [[unlikely]] {
        vengine_log("bindless allocator out or range!\n");
    }
    return ptr;
}
void DescriptorHeap::DeAllocate(luisa::FirstFit::Node *ptr) {
    std::lock_guard lck(heapMtx);
    subAlloc.free(ptr);
}
uint DescriptorHeap::GetSubAllocOffset(luisa::FirstFit::Node const *ptr) const {
    return numDescriptors - (ptr->offset() + ptr->size());
}
}// namespace lc::dx
