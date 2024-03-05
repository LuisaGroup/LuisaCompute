#pragma once
#include <Resource/Buffer.h>
#include <Resource/AllocHandle.h>
namespace lc::dx {
class DefaultBuffer final : public Buffer {
private:
    AllocHandle allocHandle;
    uint64 byteSize;
    D3D12_RESOURCE_STATES initState;
    bool _is_heap_resource = false;
    mutable std::atomic_bool _evict = false;
public:

    vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> GetColorSrvDesc(bool isRaw = false) const override;
    vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> GetColorUavDesc(bool isRaw = false) const override;
    vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> GetColorSrvDesc(uint64 offset, uint64 byteSize, bool isRaw = false) const override;
    vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> GetColorUavDesc(uint64 offset, uint64 byteSize, bool isRaw = false) const override;
    ID3D12Resource *GetResource() const override { return allocHandle.resource.Get(); }
    D3D12_GPU_VIRTUAL_ADDRESS GetAddress() const override { return allocHandle.resource->GetGPUVirtualAddress(); }
    uint64 GetByteSize() const override { return byteSize; }
    DefaultBuffer(
        Device *device,
        uint64 byteSize,
        GpuAllocator *allocator = nullptr,
        D3D12_RESOURCE_STATES initState = D3D12_RESOURCE_STATE_COMMON,
        bool shared_adaptor = false);
    ~DefaultBuffer();
    DefaultBuffer(
        Device *device,
        uint64 byteSize,
        ID3D12Resource *resource,
        D3D12_RESOURCE_STATES initState = D3D12_RESOURCE_STATE_COMMON);
    D3D12_RESOURCE_STATES GetInitState() const override {
        return initState;
    }
    Tag GetTag() const override {
        return Tag::DefaultBuffer;
    }
    bool IsHeapResource() const {
        return _is_heap_resource;
    }
    DefaultBuffer(DefaultBuffer &&rhs)
        : Buffer(std::move(rhs)),
          allocHandle(std::move(rhs.allocHandle)),
          byteSize(rhs.byteSize),
          initState(rhs.initState),
          _is_heap_resource(rhs._is_heap_resource),
          _evict(rhs._evict.load()) {}
    void Evict() const override;
    void Resident(vstd::vector<ID3D12Pageable *> &vec) const override;
    KILL_COPY_CONSTRUCT(DefaultBuffer)
};
}// namespace lc::dx
