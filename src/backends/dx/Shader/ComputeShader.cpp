#include <Shader/ComputeShader.h>
#include <Shader/ShaderSerializer.h>
#include "../../common/hlsl/hlsl_codegen.h"
#include "../../common/hlsl/shader_compiler.h"
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/md5.h>
#include "../../common/backend_print_code.h"
#include "../../common/env_flag.h"
namespace lc::dx {
class StringViewBinaryStream : public BinaryStream {

public:
    luisa::string_view strv;
    size_t _pos{};
    StringViewBinaryStream(luisa::string_view strv) : strv(strv) {}
    [[nodiscard]] size_t length() const noexcept override { return strv.size(); }
    [[nodiscard]] size_t pos() const noexcept override { return _pos; }
    void read(luisa::span<std::byte> dst) noexcept override {
        LUISA_DEBUG_ASSERT(dst.size() + _pos <= strv.size());
        std::memcpy(dst.data(), strv.data() + _pos, dst.size());
        _pos += dst.size();
    }
    ~StringViewBinaryStream() noexcept = default;
};

luisa::unique_ptr<luisa::BinaryStream> read_binary_io(CacheType type, luisa::BinaryIO const *binIo, luisa::string_view name) {
    switch (type) {
        case CacheType::ByteCode:
            return binIo->read_shader_bytecode(name);
        case CacheType::Cache:
            return binIo->read_shader_cache(name);
        case CacheType::Internal: {
            if (auto cached = binIo->read_internal_shader(name)) {
                return cached;
            }
            auto internal_data = hlsl::CodegenUtility::ReadInternalHLSLFile(name);
            if (!internal_data.empty()) {
                return luisa::make_unique<StringViewBinaryStream>(internal_data);
            }
            return {};
        }
    }
    return luisa::unique_ptr<luisa::BinaryStream>{};
}
ComputeShader *ComputeShader::load_preset_compute(
    BinaryIO const *file_io,
    luisa::compute::Profiler *profiler,
    Device *device,
    vstd::span<Type const *const> types,
    vstd::string_view fileName) {
    auto pso_name = Shader::pso_name(device, fileName);
    bool old_deleted = false;
    auto result = ShaderSerializer::DeSerialize(
        fileName,
        pso_name,
        CacheType::ByteCode,
        device,
        *file_io,
        profiler,
        {},
        hlsl::CodegenUtility::GetTypeMD5(types),
        {},
        old_deleted);
    //Cached

    if (result) {
        if (old_deleted) {
            result->_save_pso(result->pso(), pso_name, file_io, device);
        }
    }
    return result;
}
ComputeShader *ComputeShader::_compile_compute_impl(
    BinaryIO const *file_io,
    luisa::compute::Profiler *profiler,
    Device *device,
    Function kernel,
    vstd::function<hlsl::CodegenResult()> const &codegen,
    vstd::function<uint(hlsl::CodegenResult const &)> const &choose_shader_model,
    vstd::optional<vstd::MD5> const &checkMD5,
    vstd::vector<luisa::compute::Argument> &&bindings,
    uint3 blockSize,
    vstd::string_view fileName,
    CacheType cacheType,
    bool enableUnsafeMath,
    bool debug,
    vstd::optional<uint> explicit_validation_count) {
    const bool profile_compile =
        luisa::compute::detail::env_flag("LUISA_DX_PROFILE_COMPILATION");
    auto compile_new_compute = [&](bool WriteCache, vstd::string_view pso_name) {
        auto str = codegen();
        const uint shader_model = choose_shader_model(str);
        const uint effective_validation_count =
            explicit_validation_count.value_or(str.validation_count);
        vstd::MD5 md5;
        if (WriteCache) {
            if (checkMD5) {
                md5 = *checkMD5;
            } else {
                md5 = vstd::MD5({reinterpret_cast<uint8_t const *>(str.result.data() + str.immutableHeaderSize), str.result.size() - str.immutableHeaderSize});
            }
        }

        if (luisa::compute::backend_print_code_enabled()) {
            auto dump_name = [&]() -> luisa::string {
                if (!fileName.empty()) return luisa::string{fileName.data(), fileName.size()};
                if (!kernel.name().empty()) return luisa::string{kernel.name()};
                return luisa::format("{:x}", kernel.hash());
            }();
            auto dump_file_name = luisa::format("hlsl_output_{}.hlsl", dump_name);
            if (auto f = fopen(dump_file_name.c_str(), "wb")) {
                fwrite(str.result.data(), str.result.size(), 1, f);
                fclose(f);
            }
        }
        if (profiler) [[unlikely]] {
            profiler->before_compile_shader_bytecode(fileName);
        }
        Clock phase_clock;
        auto comp_result = Device::compiler()->compile_compute(
            str.result.view(),
            true,
            shader_model,
            enableUnsafeMath,
            false, debug);
        if (profile_compile) {
            auto label = (kernel.builder() != nullptr && !kernel.name().empty()) ?
                             luisa::string{kernel.name()} :
                             luisa::string{fileName};
            LUISA_INFO("DX DXC compile '{}' (sm={}, hlsl={} B): {:.3f} ms",
                       label, shader_model, str.result.size(), phase_clock.toc());
        }
        if (profiler) [[unlikely]] {
            profiler->after_compile_shader_bytecode(fileName);
        }
        return comp_result.multi_visit_or(
            vstd::UndefEval<ComputeShader *>{},
            [&](hlsl::ComUniquePtr<IDxcBlob> &buffer) {
                uint bdls_buffer_count = 0;
                if (str.useBufferBindless) bdls_buffer_count++;
                if (str.useTex2DBindless) bdls_buffer_count++;
                if (str.useTex3DBindless) bdls_buffer_count++;
                auto kernel_args = [&] {
                    if (kernel.builder() == nullptr) {
                        return vstd::vector<SavedArgument>();
                    } else {
                        return ShaderSerializer::SerializeKernel(kernel);
                    }
                }();
                if (WriteCache) {
                    auto ser_data = ShaderSerializer::Serialize(
                        str.properties,
                        kernel_args,
                        {reinterpret_cast<std::byte const *>(buffer->GetBufferPointer()),
                         buffer->GetBufferSize()},
                        md5,
                        str.typeMD5,
                        bdls_buffer_count,
                        str.validation_count,
                        blockSize,
                        str.printers);
                    write_binary_io(cacheType, file_io, fileName, {reinterpret_cast<std::byte const *>(ser_data.data()), luisa::size_bytes(ser_data)});
                }
                phase_clock.tic();
                auto cs = new ComputeShader(
                    blockSize,
                    std::move(str.properties),
                    std::move(kernel_args),
                    {reinterpret_cast<std::byte const *>(buffer->GetBufferPointer()),
                     buffer->GetBufferSize()},
                    std::move(bindings),
                    std::move(str.printers),
                    effective_validation_count,
                    device);
                if (profile_compile) {
                    auto label = (kernel.builder() != nullptr && !kernel.name().empty()) ?
                                     luisa::string{kernel.name()} :
                                     luisa::string{fileName};
                    LUISA_INFO("DX CreateComputePipelineState '{}' (dxil={} B): {:.3f} ms",
                               label, buffer->GetBufferSize(), phase_clock.toc());
                }
                cs->_bindless_count = bdls_buffer_count;
                if (WriteCache) {
                    cs->_save_pso(cs->pso(), pso_name, file_io, device);
                }
                return cs;
            },
            [](auto &&err) {
                LUISA_ERROR("Compile Error: {}", err);
                return nullptr;
            });
    };
    if (!fileName.empty()) {
        vstd::string pso_name = Shader::pso_name(device, fileName);
        bool old_deleted = false;
        Clock deser_clock;
        //Cached
        auto result = ShaderSerializer::DeSerialize(
            fileName,
            pso_name,
            cacheType,
            device,
            *file_io,
            profiler,
            checkMD5,
            {},
            std::move(bindings),
            old_deleted);
        if (result) {
            if (profile_compile) {
                auto label = (kernel.builder() != nullptr && !kernel.name().empty()) ?
                                 luisa::string{kernel.name()} :
                                 luisa::string{fileName};
                LUISA_INFO("DX cache hit '{}' ({}): {:.3f} ms",
                           label, fileName, deser_clock.toc());
            }
            if (old_deleted) {
                result->_save_pso(result->pso(), pso_name, file_io, device);
            }
            return result;
        }
        if (cacheType == CacheType::Internal) [[unlikely]] {
            LUISA_VERBOSE("Cached internal DXIL {} is missing or outdated; recompiling.", fileName);
            result = compile_new_compute(true, pso_name);
#ifndef NDEBUG
// TODO save
#endif
            return result;
        }

        return compile_new_compute(true, pso_name);
    } else {
        return compile_new_compute(false, {});
    }
}

ComputeShader *ComputeShader::compile_compute(
    BinaryIO const *file_io,
    luisa::compute::Profiler *profiler,
    Device *device,
    Function kernel,
    vstd::function<hlsl::CodegenResult()> const &codegen,
    vstd::optional<vstd::MD5> const &checkMD5,
    vstd::vector<luisa::compute::Argument> &&bindings,
    uint3 blockSize,
    uint shaderModel,
    vstd::string_view fileName,
    CacheType cacheType,
    bool enableUnsafeMath,
    bool debug,
    uint validation_count) {
    return _compile_compute_impl(
        file_io,
        profiler,
        device,
        kernel,
        codegen,
        [shaderModel](hlsl::CodegenResult const &) { return shaderModel; },
        checkMD5,
        std::move(bindings),
        blockSize,
        fileName,
        cacheType,
        enableUnsafeMath,
        debug,
        validation_count);
}
ComputeShader *ComputeShader::compile_compute(
    BinaryIO const *file_io,
    luisa::compute::Profiler *profiler,
    Device *device,
    Function kernel,
    vstd::function<hlsl::CodegenResult()> const &codegen,
    vstd::function<uint(hlsl::CodegenResult const &)> const &choose_shader_model,
    vstd::optional<vstd::MD5> const &checkMD5,
    vstd::vector<luisa::compute::Argument> &&bindings,
    uint3 blockSize,
    vstd::string_view fileName,
    CacheType cacheType,
    bool enableUnsafeMath,
    bool debug) {
    return _compile_compute_impl(
        file_io,
        profiler,
        device,
        kernel,
        codegen,
        choose_shader_model,
        checkMD5,
        std::move(bindings),
        blockSize,
        fileName,
        cacheType,
        enableUnsafeMath,
        debug,
        {});
}
bool ComputeShader::save_compute(
    BinaryIO const *file_io,
    luisa::compute::Profiler *profiler,
    Function kernel,
    hlsl::CodegenResult &str,
    uint3 blockSize,
    uint shaderModel,
    vstd::string_view fileName,
    bool enableUnsafeMath,
    bool debug) {
    vstd::MD5 md5({reinterpret_cast<uint8_t const *>(str.result.data() + str.immutableHeaderSize), str.result.size() - str.immutableHeaderSize});
    if (luisa::compute::backend_print_code_enabled()) {
        auto dump_name = [&]() -> luisa::string {
            if (!fileName.empty()) return luisa::string{fileName.data(), fileName.size()};
            if (!kernel.name().empty()) return luisa::string{kernel.name()};
            return luisa::format("{:x}", kernel.hash());
        }();
        auto dump_file_name = luisa::format("hlsl_output_{}.hlsl", dump_name);
        auto f = fopen(dump_file_name.c_str(), "wb");
        if (f) {
            fwrite(str.result.data(), str.result.size(), 1, f);
            fclose(f);
        }
    }
    if (profiler) [[unlikely]] {
        profiler->before_load_shader_bytecode(fileName);
    }
    if (ShaderSerializer::CheckMD5(fileName, md5, *file_io)) {
        LUISA_VERBOSE("Shader '{}' bytecode cache hit (MD5 match), skipping DXC recompile.", fileName);
        if (profiler) [[unlikely]] {
            profiler->after_load_shader_bytecode(fileName, true);
        }
        return true;
    } else {
        if (profiler) [[unlikely]] {
            profiler->after_load_shader_bytecode(fileName, false);
        }
    }
    auto compiler = Device::compiler();
    if (compiler) {
        if (profiler) [[unlikely]] {
            profiler->before_compile_shader_bytecode(fileName);
        }
        auto comp_result = compiler->compile_compute(
            str.result.view(),
            true,
            shaderModel,
            enableUnsafeMath,
            false, debug);
        if (profiler) [[unlikely]] {
            profiler->after_compile_shader_bytecode(fileName);
        }
        if (comp_result.is_type_of<vstd::string>()) {
            LUISA_WARNING("DXC compute-shader compile error for '{}': {}",
                           fileName, comp_result.get<1>());
            return false;
        }
        auto &buffer = comp_result.get<0>();
        auto kernel_args = ShaderSerializer::SerializeKernel(kernel);
        uint bdls_buffer_count = 0;
        if (str.useBufferBindless) bdls_buffer_count++;
        if (str.useTex2DBindless) bdls_buffer_count++;
        if (str.useTex3DBindless) bdls_buffer_count++;
        auto ser_data = ShaderSerializer::Serialize(
            str.properties,
            kernel_args,
            {reinterpret_cast<std::byte const *>(buffer->GetBufferPointer()),
             buffer->GetBufferSize()},
            md5,
            str.typeMD5,
            bdls_buffer_count,
            str.validation_count,
            blockSize,
            str.printers);
        static_cast<void>(file_io->write_shader_bytecode(fileName, {reinterpret_cast<std::byte const *>(ser_data.data()), luisa::size_bytes(ser_data)}));
    } else {
        // write HLSL code if compiler not initialized
        static_cast<void>(file_io->write_shader_bytecode(fileName, {reinterpret_cast<std::byte const *>(str.result.data()), str.result.size()}));
    }
    return true;
}
ID3D12CommandSignature *ComputeShader::cmd_sig() const {
    std::lock_guard lck(_cmd_sig_mtx);
    if (_cmd_sig) return _cmd_sig.Get();
    D3D12_COMMAND_SIGNATURE_DESC desc{};
    D3D12_INDIRECT_ARGUMENT_DESC ind_desc[2];
    std::memset(ind_desc, 0, vstd::array_byte_size(ind_desc));
    ind_desc[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    auto &c = ind_desc[0].Constant;
    c.RootParameterIndex = 0;
    c.DestOffsetIn32BitValues = 0;
    c.Num32BitValuesToSet = 4;
    ind_desc[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    desc.ByteStride = kDispatchIndirectStride;
    desc.NumArgumentDescs = 2;
    desc.pArgumentDescs = ind_desc;
    ThrowIfFailed(_device->device->CreateCommandSignature(&desc, root_sig(), IID_PPV_ARGS(&_cmd_sig)));
    return _cmd_sig.Get();
}

ComputeShader::ComputeShader(
    uint3 blockSize,
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    vstd::span<std::byte const> binData,
    vstd::vector<luisa::compute::Argument> &&bindings,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    uint validation_count,
    Device *device)
    : Shader(std::move(prop), std::move(args), device->device, std::move(printers), validation_count, false),
      _arg_bindings(std::move(bindings)),
      _device(device),
      _block_size(blockSize) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = root_sig();
    psoDesc.CS.pShaderBytecode = binData.data();
    psoDesc.CS.BytecodeLength = binData.size();
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    if (device->feature_check.cluster_acceleration_structure_supported() &&
        device->raytracing2) {
        psoDesc.Flags |= D3D12_PIPELINE_STATE_FLAG_RAYQUERY_ALLOW_CLUSTERED_GEOMETRY_ON_PRE_SM69_SHADERS;
    }
    ThrowIfFailed(device->device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(_pso.GetAddressOf())));
    _init_captured_preprocess();
}
ComputeShader::ComputeShader(
    uint3 blockSize,
    Device *device,
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    vstd::vector<luisa::compute::Argument> &&bindings,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    ComPtr<ID3D12RootSignature> &&root_sig,
    ComPtr<ID3D12PipelineState> &&pso,
    uint validation_count)
    : Shader(std::move(prop), std::move(args), std::move(root_sig), std::move(printers), validation_count),
      _arg_bindings(std::move(bindings)),
      _device(device),
      _block_size(blockSize) {
    this->_pso = std::move(pso);
    _init_captured_preprocess();
}

void ComputeShader::_init_captured_preprocess() {
    for (auto const &arg : _arg_bindings) {
        if (arg.tag == luisa::compute::Argument::Tag::UNIFORM) {
            _captured_has_uniforms = true;
            break;
        }
    }
    if (_arg_bindings.empty() || _captured_has_uniforms ||
        _arg_bindings.size() > _kernel_arguments.size()) {
        return;
    }
    bool can_fast = true;
    for (size_t i = 0u; i < _arg_bindings.size() && can_fast; ++i) {
        auto const &arg = _arg_bindings[i];
        auto const &saved = _kernel_arguments[i];
        bool const is_write =
            (static_cast<uint>(saved.var_usage) &
             static_cast<uint>(luisa::compute::Usage::WRITE)) != 0u;
        switch (arg.tag) {
            case luisa::compute::Argument::Tag::BUFFER: {
                if (arg.buffer.handle == 0u) {
                    can_fast = false;
                    break;
                }
                auto const *res =
                    reinterpret_cast<Buffer const *>(arg.buffer.handle);
                auto const tag = res->get_tag();
                if (tag != Resource::Tag::DefaultBuffer &&
                    tag != Resource::Tag::SparseBuffer &&
                    tag != Resource::Tag::ExternalBuffer) {
                    can_fast = false;
                    break;
                }
                BufferView bv{res, arg.buffer.offset, arg.buffer.size};
                if (is_write) {
                    _captured_writable_buffers.emplace_back(bv);
                } else {
                    _captured_readonly_buffers.emplace_back(bv);
                }
            } break;
            case luisa::compute::Argument::Tag::BINDLESS_ARRAY: {
                if (arg.bindless_array.handle == 0u) {
                    can_fast = false;
                    break;
                }
                _captured_bindless.emplace_back(
                    reinterpret_cast<BindlessArray *>(arg.bindless_array.handle));
            } break;
            case luisa::compute::Argument::Tag::ACCEL: {
                if (arg.accel.handle == 0u) {
                    can_fast = false;
                    break;
                }
                _captured_accels.emplace_back(arg.accel.handle, is_write);
            } break;
            default:
                can_fast = false;
                break;
        }
    }
    if (can_fast) {
        _captured_can_fast_preprocess = true;
    } else {
        _captured_readonly_buffers.clear();
        _captured_writable_buffers.clear();
        _captured_bindless.clear();
        _captured_accels.clear();
    }
}

ComputeShader::~ComputeShader() {
}
}// namespace lc::dx
