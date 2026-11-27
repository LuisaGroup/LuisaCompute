#pragma once
#include <Shader/Shader.h>
#include <luisa/core/binary_io.h>
namespace lc::hlsl {
struct CodegenResult;
}// namespace lc::hlsl
namespace lc::dx {
class ShaderSerializer;
class BindlessArray;
class ComputeShader final : public Shader {
    friend class ShaderSerializer;

private:
    ComPtr<ID3D12PipelineState> _pso;
    vstd::vector<luisa::compute::Argument> _arg_bindings;
    Device *_device;
    uint3 _block_size;
    bool _captured_has_uniforms{false};
    bool _captured_can_fast_preprocess{false};
    vstd::vector<BufferView> _captured_readonly_buffers;
    vstd::vector<BufferView> _captured_writable_buffers;
    vstd::vector<BindlessArray *> _captured_bindless;
    vstd::vector<std::pair<uint64_t, bool>> _captured_accels;
    void _init_captured_preprocess();
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
    static ComputeShader *_compile_compute_impl(
        luisa::BinaryIO const *file_io,
        luisa::compute::Profiler *profiler,
        Device *device,
        Function kernel,
        vstd::function<hlsl::CodegenResult()> const &codegen,
        vstd::function<uint(hlsl::CodegenResult const &)> const &choose_shader_model,
        vstd::optional<vstd::MD5> const &md5,
        vstd::vector<luisa::compute::Argument> &&bindings,
        uint3 blockSize,
        vstd::string_view fileName,
        CacheType cacheType,
        bool enableUnsafeMath,
        bool debug,
        vstd::optional<uint> explicit_validation_count);

public:
    static constexpr uint64_t kDispatchIndirectStride = 28;
    ID3D12PipelineState *pso() const { return _pso.Get(); }
    vstd::span<luisa::compute::Argument const> arg_bindings() const { return _arg_bindings; }
    [[nodiscard]] bool captured_has_uniforms() const noexcept { return _captured_has_uniforms; }
    [[nodiscard]] bool captured_can_fast_preprocess() const noexcept { return _captured_can_fast_preprocess; }
    [[nodiscard]] vstd::span<BufferView const> captured_readonly_buffers() const noexcept { return _captured_readonly_buffers; }
    [[nodiscard]] vstd::span<BufferView const> captured_writable_buffers() const noexcept { return _captured_writable_buffers; }
    [[nodiscard]] vstd::span<BindlessArray *const> captured_bindless() const noexcept { return _captured_bindless; }
    [[nodiscard]] vstd::span<std::pair<uint64_t, bool> const> captured_accels() const noexcept { return _captured_accels; }
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
    static ComputeShader *compile_compute(
        luisa::BinaryIO const *file_io,
        luisa::compute::Profiler *profiler,
        Device *device,
        Function kernel,
        vstd::function<hlsl::CodegenResult()> const &codegen,
        vstd::function<uint(hlsl::CodegenResult const &)> const &choose_shader_model,
        vstd::optional<vstd::MD5> const &md5,
        vstd::vector<luisa::compute::Argument> &&bindings,
        uint3 blockSize,
        vstd::string_view fileName,
        CacheType cacheType,
        bool enableUnsafeMath,
        bool debug);
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
