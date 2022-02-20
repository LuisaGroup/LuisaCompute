#pragma once
#include <Resource/Resource.h>
namespace toolhub::directx {

class DescriptorHeap final : public Resource {
private:
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> pDH;
    D3D12_DESCRIPTOR_HEAP_DESC Desc;
    D3D12_CPU_DESCRIPTOR_HANDLE hCPUHeapStart;
    D3D12_GPU_DESCRIPTOR_HANDLE hGPUHeapStart;
    uint HandleIncrementSize;
    uint64 numDescriptors;
    vstd::vector<uint> allocatePool;
    vstd::spin_mutex heapMtx;

public:
    uint64 Length() const { return numDescriptors; }
    ID3D12DescriptorHeap *GetHeap() const { return pDH.Get(); }
    D3D12_GPU_DESCRIPTOR_HANDLE hGPU(uint64 index) const {
        if (index >= Desc.NumDescriptors) index = Desc.NumDescriptors - 1;
        D3D12_GPU_DESCRIPTOR_HANDLE h = {hGPUHeapStart.ptr + index * HandleIncrementSize};
        return h;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE hCPU(uint64 index) const {
        if (index >= Desc.NumDescriptors) index = Desc.NumDescriptors - 1;
        D3D12_CPU_DESCRIPTOR_HANDLE h = {hCPUHeapStart.ptr + index * HandleIncrementSize};
        return h;
    }

    DescriptorHeap(
        Device *pDevice,
        D3D12_DESCRIPTOR_HEAP_TYPE Type,
        uint64 numDescriptors,
        bool bShaderVisible);
    uint AllocateIndex();
    void ReturnIndex(uint v);
    void Reset();
    void CreateUAV(ID3D12Resource *resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC &pDesc, uint64 index);
    void CreateSRV(ID3D12Resource *resource, const D3D12_SHADER_RESOURCE_VIEW_DESC &pDesc, uint64 index);
    void CreateSampler(D3D12_SAMPLER_DESC const &desc, uint64 index);
    ~DescriptorHeap();
    Tag GetTag() const override { return Tag::DescriptorHeap; }
    VSTD_SELF_PTR
};
struct DescriptorHeapView {
    DescriptorHeap const *heap;
    uint64 index;
    DescriptorHeapView(
        DescriptorHeap const *heap,
        uint64 index)
        : heap(heap),
          index(index) {}
    DescriptorHeapView(
        DescriptorHeap const *heap)
        : heap(heap),
          index(0) {}
};
}// namespace toolhub::directx