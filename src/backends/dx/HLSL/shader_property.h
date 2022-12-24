#pragma once
namespace toolhub::directx {

enum class ShaderVariableType : uint8_t {
    ConstantBuffer,
    SRVDescriptorHeap,
    UAVDescriptorHeap,
    CBVDescriptorHeap,
    SampDescriptorHeap,
    StructuredBuffer,
    RWStructuredBuffer,
    ConstantValue
};
struct Property {
    ShaderVariableType type;
    uint spaceIndex;
    uint registerIndex;
    uint arrSize;
};
}// namespace toolhub::directx