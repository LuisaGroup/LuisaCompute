#include <Shader/ComputeShader.h>
#include <Shader/ShaderSerializer.h>
#include <HLSL/dx_codegen.h>
#include <Shader/ShaderCompiler.h>
#include <vstl/md5.h>
namespace toolhub::directx {
namespace ComputeShaderDetail {
static constexpr bool PRINT_CODE = false;
}// namespace ComputeShaderDetail
ComputeShader *ComputeShader::LoadPresetCompute(
    BinaryIO *fileIo,
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
        false,
        device,
        *fileIo,
        {},
        typeMD5,
        oldDeleted);
    //Cached

    if (result) {
        auto md5 = CodegenUtility::GetTypeMD5(types);
        if (md5 != typeMD5) {
            LUISA_ERROR("Shader {} arguments unmatch to requirement!", fileName);
        }
        if (oldDeleted) {
            result->SavePSO(psoName, fileIo, device);
        }
    }
    return result;
}
ComputeShader *ComputeShader::CompileCompute(
    BinaryIO *fileIo,
    Device *device,
    Function kernel,
    vstd::function<CodegenResult()> const &codegen,
    vstd::optional<vstd::MD5> const &checkMD5,
    uint3 blockSize,
    uint shaderModel,
    vstd::string_view fileName,
    bool isInternal) {

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

        if constexpr (PRINT_CODE) {
            auto f = fopen("hlsl_output.hlsl", "ab");
            fwrite(str.result.data(), str.result.size(), 1, f);
            fclose(f);
        }
        auto compResult = Device::Compiler()->CompileCompute(
            str.result.view(),
            true,
            shaderModel);
        return compResult.multi_visit_or(
            vstd::UndefEval<ComputeShader *>{},
            [&](vstd::unique_ptr<DXByteBlob> const &buffer) {
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
                        {buffer->GetBufferPtr(), buffer->GetBufferSize()},
                        md5,
                        str.typeMD5,
                        str.bdlsBufferCount,
                        blockSize);
                    if (isInternal) {
                        fileIo->write_internal_shader(fileName, {reinterpret_cast<std::byte const *>(serData.data()), serData.size_bytes()});
                    } else {
                        fileIo->write_shader_bytecode(fileName, {reinterpret_cast<std::byte const *>(serData.data()), serData.size_bytes()});
                    }
                }
                auto cs = new ComputeShader(
                    blockSize,
                    std::move(str.properties),
                    std::move(kernelArgs),
                    {buffer->GetBufferPtr(),
                     buffer->GetBufferSize()},
                    device);
                cs->bindlessCount = str.bdlsBufferCount;
                if (WriteCache) {
                    cs->SavePSO(psoName, fileIo, device);
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
            isInternal,
            device,
            *fileIo,
            checkMD5,
            typeMD5,
            oldDeleted);
        if (result) {
            if (oldDeleted) {
                result->SavePSO(psoName, fileIo, device);
            }
            return result;
        }

        return CompileNewCompute(true, psoName);
    } else {
        return CompileNewCompute(false, {});
    }
}
void ComputeShader::SaveCompute(
    BinaryIO *fileIo,
    Function kernel,
    CodegenResult &str,
    uint3 blockSize,
    uint shaderModel,
    vstd::string_view fileName) {
    using namespace ComputeShaderDetail;
    vstd::MD5 md5({reinterpret_cast<uint8_t const *>(str.result.data() + str.immutableHeaderSize), str.result.size() - str.immutableHeaderSize});
    if constexpr (PRINT_CODE) {
        auto f = fopen("hlsl_output.hlsl", "ab");
        fwrite(str.result.data(), str.result.size(), 1, f);
        fclose(f);
    }
    if (ShaderSerializer::CheckMD5(fileName, md5, *fileIo)) return;
    auto compResult = Device::Compiler()->CompileCompute(
        str.result.view(),
        true,
        shaderModel);
    compResult.multi_visit(
        [&](vstd::unique_ptr<DXByteBlob> const &buffer) {
            auto kernelArgs = ShaderSerializer::SerializeKernel(kernel);
            auto serData = ShaderSerializer::Serialize(
                str.properties,
                kernelArgs,
                {buffer->GetBufferPtr(), buffer->GetBufferSize()},
                md5,
                str.typeMD5,
                str.bdlsBufferCount,
                blockSize);
            fileIo->write_shader_bytecode(fileName, {reinterpret_cast<std::byte const *>(serData.data()), serData.size_bytes()});
        },
        [](auto &&err) {
            std::cout << err << '\n';
            VSTL_ABORT();
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
    c.RootParameterIndex = 1;
    c.DestOffsetIn32BitValues = 0;
    c.Num32BitValuesToSet = 4;
    indDesc[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    desc.ByteStride = 28;
    desc.NumArgumentDescs = 2;
    desc.pArgumentDescs = indDesc;
    ThrowIfFailed(device->device->CreateCommandSignature(&desc, rootSig.Get(), IID_PPV_ARGS(&cmdSig)));
    return cmdSig.Get();
}

ComputeShader::ComputeShader(
    uint3 blockSize,
    vstd::vector<Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    vstd::span<std::byte const> binData,
    Device *device)
    : Shader(std::move(prop), std::move(args), device->device, false),
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
    vstd::vector<Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ComPtr<ID3D12RootSignature> &&rootSig,
    ComPtr<ID3D12PipelineState> &&pso)
    : Shader(std::move(prop), std::move(args), std::move(rootSig)),
      device(device),
      blockSize(blockSize) {
    this->pso = std::move(pso);
}

ComputeShader::~ComputeShader() {
}
}// namespace toolhub::directx