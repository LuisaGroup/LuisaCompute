#pragma once
#include <Shader/Shader.h>
#include <luisa/core/binary_io.h>
namespace lc::hlsl {
struct CodegenResult;
}// namespace lc::hlsl
namespace lc::dx {
class ShaderSerializer;
class ComputeShader final : public Shader {
    friend class ShaderSerializer;

private:
    ComPtr<ID3D12PipelineState> _pso;
    vstd::vector<luisa::compute::Argument> _arg_bindings;
    Device *_device;
    uint3 _block_size;
    ComputeShader(
        uint3 blockSize,
        Device *device,
        vstd::vector<hlsl::Property> &&prop,
        vstd::vector<SavedArgument> &&args,
        vstd::vector<luisa::compute::Argument> &&bindings,
        vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
        ComPtr<ID3D12RootSignature> &&rootSig,
        ComPtr<ID3D12PipelineState> &&pso,
        uint validation_count);

    mutable ComPtr<ID3D12CommandSignature> _cmd_sig;
    mutable std::mutex _cmd_sig_mtx;

public:
    static constexpr uint64_t kDispatchIndirectStride = 28;
    ID3D12PipelineState *pso() const { return _pso.Get(); }
    vstd::span<luisa::compute::Argument const> arg_bindings() const { return _arg_bindings; }
    ID3D12CommandSignature *cmd_sig() const;
    Device *get_device() const { return _device; }
    Tag get_tag() const { return Tag::ComputeShader; }
    uint3 block_size() const { return _block_size; }
    static ComputeShader *compile_compute(
        luisa::BinaryIO const *file_io,
        luisa::compute::Profiler *profiler,
        Device *device,
        Function kernel,
        vstd::function<hlsl::CodegenResult()> const &codegen,
        vstd::optional<vstd::MD5> const &md5,
        vstd::vector<luisa::compute::Argument> &&bindings,
        uint3 blockSize,
        uint shaderModel,
        vstd::string_view fileName,
        CacheType cacheType,
        bool enableUnsafeMath,
        bool debug,
        uint validation_count);
    // Returns true if the bytecode file was written (or cache hit).
    // Returns false if DXC compilation failed — the caller can continue
    // instead of being killed by std::abort (which LUISA_ERROR does).
    static bool save_compute(
        luisa::BinaryIO const *file_io,
        luisa::compute::Profiler *profiler,
        Function kernel,
        hlsl::CodegenResult &codegen,
        uint3 blockSize,
        uint shaderModel,
        vstd::string_view fileName,
        bool enableUnsafeMath,
        bool debug);
    static ComputeShader *load_preset_compute(
        luisa::BinaryIO const *file_io,
        luisa::compute::Profiler *profiler,
        Device *device,
        vstd::span<Type const *const> types,
        vstd::string_view fileName);
    ComputeShader(
        uint3 blockSize,
        vstd::vector<hlsl::Property> &&properties,
        vstd::vector<SavedArgument> &&args,
        vstd::span<std::byte const> binData,
        vstd::vector<luisa::compute::Argument> &&bindings,
        vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
        uint validation_count,
        Device *device);
    ~ComputeShader();
    KILL_COPY_CONSTRUCT(ComputeShader)
    KILL_MOVE_CONSTRUCT(ComputeShader)
};
}// namespace lc::dx
