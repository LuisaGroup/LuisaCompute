#include <Resource/ExternalBuffer.h>
namespace toolhub::directx {
ExternalBuffer::ExternalBuffer(Device *device, ID3D12Resource *resource, D3D12_RESOURCE_STATES initState)
    : Buffer(device), resource(resource), initState(initState) {
    auto desc = resource->GetDesc();
    byteSize = desc.Width;
}
vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> ExternalBuffer::GetColorSrvDesc(bool isRaw) const {
    return GetColorSrvDesc(0, byteSize, isRaw);
}
vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> ExternalBuffer::GetColorUavDesc(bool isRaw) const {
    return GetColorUavDesc(0, byteSize, isRaw);
}
vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> ExternalBuffer::GetColorSrvDesc(uint64 offset, uint64 byteSize, bool isRaw) const {
    D3D12_SHADER_RESOURCE_VIEW_DESC res;
    res.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    res.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (isRaw) {
        res.Format = DXGI_FORMAT_R32_TYPELESS;
        assert((offset & 15) == 0);
        res.Buffer.FirstElement = offset / 4;
        res.Buffer.NumElements = byteSize / 4;
        res.Buffer.StructureByteStride = 0;
        res.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    } else {
        res.Format = DXGI_FORMAT_UNKNOWN;
        assert((offset & 3) == 0);
        res.Buffer.FirstElement = offset / 4;
        res.Buffer.NumElements = byteSize / 4;
        res.Buffer.StructureByteStride = 4;
        res.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    }
    return res;
}
vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> ExternalBuffer::GetColorUavDesc(uint64 offset, uint64 byteSize, bool isRaw) const {
    D3D12_UNORDERED_ACCESS_VIEW_DESC res;
    res.Format = DXGI_FORMAT_R32_TYPELESS;
    res.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    res.Buffer.CounterOffsetInBytes = 0;
    if (isRaw) {
        assert((offset & 15) == 0);
        res.Buffer.FirstElement = offset / 4;
        res.Buffer.NumElements = byteSize / 4;
        res.Buffer.StructureByteStride = 0;
        res.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        assert((offset & 3) == 0);
        res.Buffer.FirstElement = offset / 4;
        res.Buffer.NumElements = byteSize / 4;
        res.Buffer.StructureByteStride = 4;
        res.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    }
    return res;
}
}// namespace toolhub::directx