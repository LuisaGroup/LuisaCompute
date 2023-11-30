#include <Shader/ComputeShader.h>
#include <Shader/ShaderSerializer.h>
#include "../../common/hlsl/hlsl_codegen.h"
#include "../../common/hlsl/shader_compiler.h"
#include <luisa/core/logging.h>
#include <luisa/vstl/md5.h>
namespace lc::dx {
namespace ComputeShaderDetail {
static const bool PRINT_CODE = ([] {
    // read env LUISA_DUMP_SOURCE
    auto env = std::getenv("LUISA_DUMP_SOURCE");
    if (env == nullptr) return false;
    return std::string_view{env} == "1";
})();
}// namespace ComputeShaderDetail
ComputeShader *ComputeShader::LoadPresetCompute(
    BinaryIO const *fileIo,
    Device *device,
    vstd::span<Type const *const> types,
    vstd::string_view fileName) {
    using namespace ComputeShaderDetail;
    auto psoName = Shader::PSOName(device, fileName);
    bool oldDeleted = false;
    vstd::MD5 typeMD5;
    auto result = ShaderSerializer::DeSerialize(
        fileName,
        psoName,
        CacheType::ByteCode,
        device,
        *fileIo,
        {},
        typeMD5,
        {},
        oldDeleted);
    //Cached

    if (result) {
        auto md5 = hlsl::CodegenUtility::GetTypeMD5(types);
        LUISA_ASSERT(md5 == typeMD5, "Shader {} arguments unmatch to requirement!", fileName);
        if (oldDeleted) {
            result->SavePSO(result->Pso(), psoName, fileIo, device);
        }
    }
    return result;
}
ComputeShader *ComputeShader::CompileCompute(
    BinaryIO const *fileIo,
    Device *device,
    Function kernel,
    vstd::function<hlsl::CodegenResult()> const &codegen,
    vstd::optional<vstd::MD5> const &checkMD5,
    vstd::vector<luisa::compute::Argument> &&bindings,
    uint3 blockSize,
    uint shaderModel,
    vstd::string_view fileName,
    CacheType cacheType,
    bool enableUnsafeMath) {

    using namespace ComputeShaderDetail;
    auto CompileNewCompute = [&](bool WriteCache, vstd::string_view psoName) {
        auto str = codegen();
        vstd::MD5 md5;
        if (WriteCache) {
            if (checkMD5) {
                md5 = *checkMD5;
            } else {
                md5 = vstd::MD5({reinterpret_cast<uint8_t const *>(str.result.data() + str.immutableHeaderSize), str.result.size() - str.immutableHeaderSize});
            }
        }

        if (PRINT_CODE) {
            auto md5_str = md5.to_string();
            auto dump_file_name = vstd::string("hlsl_output_") + md5_str + ".hlsl";
            if (auto f = fopen(dump_file_name.c_str(), "wb")) {
                fwrite(str.result.data(), str.result.size(), 1, f);
                fclose(f);
            }
        }
        auto compResult = Device::Compiler()->compile_compute(
            str.result.view(),
            true,
            shaderModel,
            enableUnsafeMath,
            false);
        return compResult.multi_visit_or(
            vstd::UndefEval<ComputeShader *>{},
            [&](vstd::unique_ptr<hlsl::DxcByteBlob> const &buffer) {
                uint bdlsBufferCount = 0;
                if (str.useBufferBindless) bdlsBufferCount++;
                if (str.useTex2DBindless) bdlsBufferCount++;
                if (str.useTex3DBindless) bdlsBufferCount++;
                auto kernelArgs = [&] {
                    if (kernel.builder() == nullptr) {
                        return vstd::vector<SavedArgument>();
                    } else {
                        return ShaderSerializer::SerializeKernel(kernel);
                    }
                }();
                if (WriteCache) {
                    auto serData = ShaderSerializer::Serialize(
                        str.properties,
                        kernelArgs,
                        {buffer->data(), buffer->size()},
                        md5,
                        str.typeMD5,
                        bdlsBufferCount,
                        blockSize,
                        str.printers);
                    WriteBinaryIO(cacheType, fileIo, fileName, {reinterpret_cast<std::byte const *>(serData.data()), serData.size_bytes()});
                }
                auto cs = new ComputeShader(
                    blockSize,
                    std::move(str.properties),
                    std::move(kernelArgs),
                    {buffer->data(),
                     buffer->size()},
                    std::move(bindings),
                    std::move(str.printers),
                    device);
                cs->bindlessCount = bdlsBufferCount;
                if (WriteCache) {
                    cs->SavePSO(cs->Pso(), psoName, fileIo, device);
                }
                return cs;
            },
            [](auto &&err) {
                LUISA_ERROR("Compile Error: {}", err);
                return nullptr;
            });
    };
    if (!fileName.empty()) {
        vstd::string psoName = Shader::PSOName(device, fileName);
        bool oldDeleted = false;
        vstd::MD5 typeMD5;
        //Cached
        auto result = ShaderSerializer::DeSerialize(
            fileName,
            psoName,
            cacheType,
            device,
            *fileIo,
            checkMD5,
            typeMD5,
            std::move(bindings),
            oldDeleted);
        if (result) {
            if (oldDeleted) {
                result->SavePSO(result->Pso(), psoName, fileIo, device);
            }
            return result;
        }

        return CompileNewCompute(true, psoName);
    } else {
        return CompileNewCompute(false, {});
    }
}
void ComputeShader::SaveCompute(
    BinaryIO const *fileIo,
    Function kernel,
    hlsl::CodegenResult &str,
    uint3 blockSize,
    uint shaderModel,
    vstd::string_view fileName,
    bool enableUnsafeMath) {
    using namespace ComputeShaderDetail;
    vstd::MD5 md5({reinterpret_cast<uint8_t const *>(str.result.data() + str.immutableHeaderSize), str.result.size() - str.immutableHeaderSize});
    if (PRINT_CODE) {
        auto f = fopen("hlsl_output.hlsl", "ab");
        fwrite(str.result.data(), str.result.size(), 1, f);
        fclose(f);
    }
    if (ShaderSerializer::CheckMD5(fileName, md5, *fileIo)) return;
    auto compResult = Device::Compiler()->compile_compute(
        str.result.view(),
        true,
        shaderModel,
        enableUnsafeMath,
        false);
    compResult.multi_visit(
        [&](vstd::unique_ptr<hlsl::DxcByteBlob> const &buffer) {
            auto kernelArgs = ShaderSerializer::SerializeKernel(kernel);
            uint bdlsBufferCount = 0;
            if (str.useBufferBindless) bdlsBufferCount++;
            if (str.useTex2DBindless) bdlsBufferCount++;
            if (str.useTex3DBindless) bdlsBufferCount++;
            auto serData = ShaderSerializer::Serialize(
                str.properties,
                kernelArgs,
                {buffer->data(), buffer->size()},
                md5,
                str.typeMD5,
                bdlsBufferCount,
                blockSize,
                str.printers);
            static_cast<void>(fileIo->write_shader_bytecode(fileName, {reinterpret_cast<std::byte const *>(serData.data()), serData.size_bytes()}));
        },
        [](auto &&err) {
            LUISA_ERROR("DXC compute-shader compile error: {}", err);
        });
}
ID3D12CommandSignature *ComputeShader::CmdSig() const {
    std::lock_guard lck(cmdSigMtx);
    if (cmdSig) return cmdSig.Get();
    D3D12_COMMAND_SIGNATURE_DESC desc{};
    D3D12_INDIRECT_ARGUMENT_DESC indDesc[2];
    memset(indDesc, 0, vstd::array_byte_size(indDesc));
    indDesc[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    auto &c = indDesc[0].Constant;
    c.RootParameterIndex = 0;
    c.DestOffsetIn32BitValues = 0;
    c.Num32BitValuesToSet = 4;
    indDesc[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    desc.ByteStride = DispatchIndirectStride;
    desc.NumArgumentDescs = 2;
    desc.pArgumentDescs = indDesc;
    ThrowIfFailed(device->device->CreateCommandSignature(&desc, rootSig.Get(), IID_PPV_ARGS(&cmdSig)));
    return cmdSig.Get();
}

ComputeShader::ComputeShader(
    uint3 blockSize,
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    vstd::span<std::byte const> binData,
    vstd::vector<luisa::compute::Argument> &&bindings,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    Device *device)
    : Shader(std::move(prop), std::move(args), device->device, std::move(printers), false),
      argBindings(std::move(bindings)),
      device(device),
      blockSize(blockSize) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = rootSig.Get();
    psoDesc.CS.pShaderBytecode = binData.data();
    psoDesc.CS.BytecodeLength = binData.size();
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    ThrowIfFailed(device->device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso.GetAddressOf())));
}
ComputeShader::ComputeShader(
    uint3 blockSize,
    Device *device,
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    vstd::vector<luisa::compute::Argument> &&bindings,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    ComPtr<ID3D12RootSignature> &&rootSig,
    ComPtr<ID3D12PipelineState> &&pso)
    : Shader(std::move(prop), std::move(args), std::move(rootSig), std::move(printers)),
      argBindings(std::move(bindings)),
      device(device),
      blockSize(blockSize) {
    this->pso = std::move(pso);
}

ComputeShader::~ComputeShader() {
}
}// namespace lc::dx
