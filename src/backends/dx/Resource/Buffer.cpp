#include <Resource/Buffer.h>
namespace lc::dx {
BufferView::BufferView(Buffer const *buffer)
    : buffer(buffer),
      offset(0),
      byteSize(buffer ? buffer->GetByteSize() : 0) {
}
BufferView::BufferView(
    Buffer const *buffer,
    uint64 offset)
    : buffer(buffer),
      offset(offset) {
    byteSize = buffer->GetByteSize() - offset;
}
Buffer::Buffer(
    Device *device)
    : Resource(device) {
}
Buffer::~Buffer() {
}
D3D12_SHADER_RESOURCE_VIEW_DESC Buffer::GetColorSrvDescBase(uint64 offset, uint64 byteSize, bool isRaw) const {
    D3D12_SHADER_RESOURCE_VIEW_DESC res;
    res.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    res.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (isRaw) {
        res.Format = DXGI_FORMAT_R32_TYPELESS;
        LUISA_ASSUME((offset & 15) == 0);
        res.Buffer.FirstElement = offset / 4;
        res.Buffer.NumElements = std::min<size_t>(byteSize / 4, 1073741536 - res.Buffer.FirstElement);
        res.Buffer.StructureByteStride = 0;
        res.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    } else {
        res.Format = DXGI_FORMAT_UNKNOWN;
        LUISA_ASSUME((offset & 3) == 0);
        res.Buffer.FirstElement = offset / 4;
        res.Buffer.NumElements = std::min<size_t>(byteSize / 4, 1073741536 - res.Buffer.FirstElement);
        res.Buffer.StructureByteStride = 4;
        res.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    }
    return res;
}
D3D12_UNORDERED_ACCESS_VIEW_DESC Buffer::GetColorUavDescBase(uint64 offset, uint64 byteSize, bool isRaw) const {
    D3D12_UNORDERED_ACCESS_VIEW_DESC res;
    res.Format = DXGI_FORMAT_R32_TYPELESS;
    res.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    res.Buffer.CounterOffsetInBytes = 0;
    if (isRaw) {
        LUISA_ASSUME((offset & 15) == 0);
        res.Buffer.FirstElement = offset / 4;
        res.Buffer.NumElements = std::min<size_t>(byteSize / 4, 1073741536 - res.Buffer.FirstElement);
        res.Buffer.StructureByteStride = 0;
        res.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        LUISA_ASSUME((offset & 3) == 0);
        res.Buffer.FirstElement = offset / 4;
        res.Buffer.NumElements = std::min<size_t>(byteSize / 4, 1073741536 - res.Buffer.FirstElement);
        res.Buffer.StructureByteStride = 4;
        res.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    }
    return res;
}
}// namespace lc::dx
