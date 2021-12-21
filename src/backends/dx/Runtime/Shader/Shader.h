#pragma once
#include <vstl/Common.h>
#include <Windows.h>
#include <d3dx12.h>
#include <Shader/ShaderVariableType.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
namespace toolhub::directx {
class TopAccel;
class CommandBufferBuilder;
class Shader : public vstd::IOperatorNewBase {
protected:
    struct Property {
        ShaderVariableType type;
        uint spaceIndex;
        uint registerIndex;
        uint arrSize;
        uint rootSigPos;
    };
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
    vstd::HashMap<vstd::string, Property> properties;
    vstd::optional<Property> GetProperty(vstd::string_view str);

public:
    Shader(
        std::span<std::pair<vstd::string_view, Property>> properties,
        ID3D12Device *device);
    Shader(Shader &&v) = default;
    ID3D12RootSignature *RootSig() const { return rootSig.Get(); }
    bool SetComputeResource(
        vstd::string_view propertyName,
        CommandBufferBuilder *cmdList,
        BufferView buffer);
    bool SetComputeResource(
        vstd::string_view propertyName,
        CommandBufferBuilder *cmdList,
        DescriptorHeapView view);
    bool SetComputeResource(
        vstd::string_view propertyName,
        CommandBufferBuilder *cmdList,
        TopAccel const *bAccel);

    KILL_COPY_CONSTRUCT(Shader)
    virtual ~Shader() = default;
};
}// namespace toolhub::directx