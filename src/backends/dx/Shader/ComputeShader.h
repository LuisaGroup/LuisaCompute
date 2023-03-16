#pragma once
#include <Shader/Shader.h>
#include <core/binary_io.h>
namespace toolhub::directx {
struct CodegenResult;
class ShaderSerializer;
class ComputeShader final : public Shader {
    friend class ShaderSerializer;

private:
    Device *device;
    uint3 blockSize;
    ComputeShader(
        uint3 blockSize,
        Device *device,
        vstd::vector<Property> &&prop,
        vstd::vector<SavedArgument> &&args,
        ComPtr<ID3D12RootSignature> &&rootSig,
        ComPtr<ID3D12PipelineState> &&pso);

    mutable ComPtr<ID3D12CommandSignature> cmdSig;
    mutable std::mutex cmdSigMtx;

public:
    ID3D12CommandSignature* CmdSig() const;
    Device *GetDevice() const { return device; }
    Tag GetTag() const { return Tag::ComputeShader; }
    uint3 BlockSize() const { return blockSize; }
    static ComputeShader *CompileCompute(
        luisa::BinaryIO const*fileIo,
        Device *device,
        Function kernel,
        vstd::function<CodegenResult()> const &codegen,
        vstd::optional<vstd::MD5> const &md5,
        uint3 blockSize,
        uint shaderModel,
        vstd::string_view fileName,
        bool isInternal);
    static void SaveCompute(
        luisa::BinaryIO const*fileIo,
        Function kernel,
        CodegenResult &codegen,
        uint3 blockSize,
        uint shaderModel,
        vstd::string_view fileName);
    static ComputeShader *LoadPresetCompute(
        luisa::BinaryIO const*fileIo,
        Device *device,
        vstd::span<Type const *const> types,
        vstd::string_view fileName);
    ComputeShader(
        uint3 blockSize,
        vstd::vector<Property> &&properties,
        vstd::vector<SavedArgument> &&args,
        vstd::span<std::byte const> binData,
        Device *device);
    ~ComputeShader();
    KILL_COPY_CONSTRUCT(ComputeShader)
    KILL_MOVE_CONSTRUCT(ComputeShader)
};
}// namespace toolhub::directx