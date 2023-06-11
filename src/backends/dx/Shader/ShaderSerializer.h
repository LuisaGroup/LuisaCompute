#pragma once
#include <d3dx12.h>
#include <Shader/Shader.h>
#include <luisa/core/binary_io.h>
#include <luisa/runtime/raster/raster_state.h>
namespace lc::dx {
class ComputeShader;
class RasterShader;
class RTShader;
struct ShaderBuildData {
    vstd::vector<std::byte> binData;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig;
};

class ShaderSerializer {
    static size_t SerializeRootSig(
        vstd::span<hlsl::Property const> properties,
        vstd::vector<std::byte> &result,
        bool isRasterShader);
    static ComPtr<ID3D12RootSignature> DeSerializeRootSig(
        ID3D12Device *device,
        vstd::span<std::byte const> bytes);

public:
    static ComPtr<ID3DBlob> SerializeRootSig(vstd::span<hlsl::Property const> properties, bool isRasterShader);
    static vstd::vector<std::byte> Serialize(
        vstd::span<hlsl::Property const> properties,
        vstd::span<SavedArgument const> kernelArgs,
        vstd::span<std::byte const> binByte,
        vstd::MD5 const &checkMD5,
        vstd::MD5 const &typeMD5,
        uint bindlessCount,
        uint3 blockSize);
    static vstd::vector<std::byte> RasterSerialize(
        vstd::span<hlsl::Property const> properties,
        vstd::span<SavedArgument const> kernelArgs,
        vstd::span<std::byte const> vertBin,
        vstd::span<std::byte const> pixelBin,
        vstd::MD5 const &checkMD5,
        vstd::MD5 const &typeMD5,
        uint bindlessCount);
    static ComputeShader *DeSerialize(
        luisa::string_view fileName,
        luisa::string_view psoName,
        CacheType cacheType,
        Device *device,
        luisa::BinaryIO const &streamFunc,
        vstd::optional<vstd::MD5> const &checkMD5,
        vstd::MD5 &typeMD5,
        vstd::vector<luisa::compute::Argument> &&bindings,
        bool &clearCache);
    static RasterShader *RasterDeSerialize(
        luisa::string_view fileName,
        CacheType cacheType,
        Device *device,
        luisa::BinaryIO const &streamFunc,
        vstd::optional<vstd::MD5> const &ilMd5,
        vstd::MD5 &typeMD5,
        MeshFormat const &meshFormat);
    static bool CheckMD5(
        vstd::string_view fileName,
        vstd::MD5 const &checkMD5,
        luisa::BinaryIO const &streamFunc);
    static vstd::vector<SavedArgument> SerializeKernel(Function kernel);
    static vstd::vector<SavedArgument> SerializeKernel(
        vstd::IRange<std::pair<Variable, Usage>> &arguments);
    ShaderSerializer() = delete;
    ~ShaderSerializer() = delete;
};
}// namespace lc::dx
