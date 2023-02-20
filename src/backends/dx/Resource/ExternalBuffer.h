#pragma once
#include <Resource/Buffer.h>
namespace toolhub::directx {
class ExternalBuffer final : public Buffer {
private:
    ID3D12Resource *resource;
    uint64 byteSize;
    D3D12_RESOURCE_STATES initState;

public:
    ExternalBuffer(Device* device, ID3D12Resource *resource, D3D12_RESOURCE_STATES initState);
    vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> GetColorSrvDesc(bool isRaw = false) const override;
    vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> GetColorUavDesc(bool isRaw = false) const override;
    vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> GetColorSrvDesc(uint64 offset, uint64 byteSize, bool isRaw = false) const override;
    vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> GetColorUavDesc(uint64 offset, uint64 byteSize, bool isRaw = false) const override;
    ID3D12Resource *GetResource() const override { return resource; }
    D3D12_GPU_VIRTUAL_ADDRESS GetAddress() const override { return resource->GetGPUVirtualAddress(); }
    uint64 GetByteSize() const override { return byteSize; }
    D3D12_RESOURCE_STATES GetInitState() const override {
        return initState;
    }
    Tag GetTag() const override {
        return Tag::ExternalBuffer;
    }
    KILL_COPY_CONSTRUCT(ExternalBuffer)
    VSTD_SELF_PTR
};
}// namespace toolhub::directx