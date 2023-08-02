#pragma once
#include <Resource/Resource.h>
#include <Resource/BufferView.h>
namespace lc::dx {
class Buffer : public Resource {
protected:
    D3D12_SHADER_RESOURCE_VIEW_DESC GetColorSrvDescBase(uint64 offset, uint64 byteSize, bool isRaw) const;
    D3D12_UNORDERED_ACCESS_VIEW_DESC GetColorUavDescBase(uint64 offset, uint64 byteSize, bool isRaw) const;

public:
    Buffer(Device *device);
    virtual vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> GetColorSrvDesc(bool isRaw = false) const { return {}; }
    virtual vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> GetColorUavDesc(bool isRaw = false) const { return {}; }
    virtual vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> GetColorSrvDesc(uint64 offset, uint64 byteSize, bool isRaw = false) const { return {}; }
    virtual vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> GetColorUavDesc(uint64 offset, uint64 byteSize, bool isRaw = false) const { return {}; }
    virtual D3D12_GPU_VIRTUAL_ADDRESS GetAddress() const = 0;
    virtual uint64 GetByteSize() const = 0;
    virtual ~Buffer() override;
    Buffer(Buffer &&) = default;
    KILL_COPY_CONSTRUCT(Buffer)
};
}// namespace lc::dx
