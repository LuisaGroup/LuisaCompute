#pragma once
#include <d3dx12.h>
#include <Shader/Shader.h>
#include <vstl/MD5.h>
namespace toolhub::directx {
class ComputeShader;
class RTShader;
struct ShaderBuildData {
    vstd::vector<vbyte> binData;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
};
class ShaderSerializer {
    static size_t SerializeRootSig(
        vstd::span<std::pair<vstd::string, Shader::Property> const> properties,
        vstd::vector<vbyte> &result);
    static size_t SerializeReflection(
        vstd::span<std::pair<vstd::string, Shader::Property> const> properties,
        vstd::vector<vbyte> &result);
    static vstd::vector<std::pair<vstd::string, Shader::Property>> DeSerializeReflection(
        vstd::span<vbyte const> bytes);
    static ComPtr<ID3D12RootSignature> DeSerializeRootSig(
        ID3D12Device *device,
        vstd::span<vbyte const> bytes);

public:
    static vstd::vector<vbyte>
    Serialize(
        vstd::span<std::pair<vstd::string, Shader::Property> const> properties,
        vstd::span<vbyte> binByte,
        Shader::Tag tag,
        bool useTraceClosest,
        uint3 blockSize);
    static vstd::variant<
        ComputeShader *,
        RTShader *>
    DeSerialize(
        Device *device,
        vstd::MD5 md5,
        vstd::span<vbyte const> data);
    ShaderSerializer() = delete;
    ~ShaderSerializer() = delete;
};
}// namespace toolhub::directx