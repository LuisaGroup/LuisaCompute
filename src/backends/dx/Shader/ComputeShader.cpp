#pragma vengine_package vengine_directx
#include <Shader/ComputeShader.h>
#include <Shader/ShaderSerializer.h>
#include <vstl/BinaryReader.h>
#include <Codegen/ShaderHeader.h>
#include <Codegen/DxCodegen.h>
#include <Shader/ShaderCompiler.h>
#include <vstl/MD5.h>
namespace toolhub::directx {
ComputeShader *ComputeShader::CompileCompute(
    Device *device,
    CodegenResult &str,
    uint3 blockSize,
    uint shaderModel) {
    struct SerializeVisitor : ShaderSerializer::Visitor {
        BinaryReader csoReader;
        vstd::string const &psoPath;
        bool oldDeleted = false;
        SerializeVisitor(
            vstd::string const &path,
            vstd::string const &psoPath)
            : csoReader(path),
              psoPath(psoPath) {
        }
        vstd::vector<vbyte> readCache;
        vbyte const *ReadFile(size_t size) override {
            readCache.clear();
            readCache.resize(size);
            csoReader.Read(reinterpret_cast<char *>(readCache.data()), size);
            return readCache.data();
        }
        ShaderSerializer::ReadResult ReadFileAndPSO(
            size_t fileSize) override {
            BinaryReader psoReader(psoPath);
            ShaderSerializer::ReadResult result;
            if (psoReader) {
                size_t psoSize = psoReader.GetLength();
                readCache.resize(psoSize + fileSize);
                result.fileSize = fileSize;
                result.fileData = readCache.data();
                result.psoSize = psoSize;
                result.psoData = readCache.data() + fileSize;
                csoReader.Read(reinterpret_cast<char *>(readCache.data()), fileSize);
                psoReader.Read(reinterpret_cast<char *>(readCache.data() + fileSize), psoSize);
            } else {
                oldDeleted = true;
                readCache.resize(fileSize);
                result.fileSize = fileSize;
                result.fileData = readCache.data();
                result.psoSize = 0;
                result.psoData = nullptr;
                csoReader.Read(reinterpret_cast<char *>(readCache.data()), fileSize);
            }
            return result;
        }
        void DeletePSOFile() override {
            oldDeleted = true;
        }
    };
    static DXShaderCompiler dxCompiler;
    auto md5 = vstd::MD5(vstd::span<vbyte const>((vbyte const *)str.result.data(), str.result.size()));
    vstd::string path;
    vstd::string psoPath;
    auto savePso = [&](ComputeShader const *cs) {
        auto f = fopen(psoPath.c_str(), "wb");
        if (f) {
            auto disp = vstd::create_disposer([&] { fclose(f); });
            ComPtr<ID3DBlob> psoCache;
            cs->Pso()->GetCachedBlob(&psoCache);
            fwrite(psoCache->GetBufferPointer(), psoCache->GetBufferSize(), 1, f);
        }
    };
    path << ".cache/" << md5.ToString();
    psoPath = path;
    path << ".cso";
    psoPath << ".pso";
    static constexpr bool USE_CACHE = 0;
    if constexpr (USE_CACHE) {
        SerializeVisitor visitor(
            path,
            psoPath);

        //Cached
        if (visitor.csoReader) {
            auto result = ShaderSerializer::DeSerialize(
                str.properties,
                device,
                md5,
                visitor);   
            if (result) {
                //std::cout << "Read cache success!"sv << '\n';
                if (visitor.oldDeleted) {
                    savePso(result);
                }
                return result;
            }
        }
    }
    // Not Cached
    vstd::string compileString(GetHLSLHeader());  
    auto compResult = [&] {
        compileString << str.result;
        std::cout
            << "\n===============================\n"
            << compileString
            << "\n===============================\n";
        return dxCompiler.CompileCompute(
            compileString,
            true,
            shaderModel);
    }();
    str.properties.emplace_back(
        "samplers"sv,
        Shader::Property{
            ShaderVariableType::SampDescriptorHeap,
            1u,
            0u,
            16u});
    return compResult.multi_visit_or(
        (ComputeShader *)nullptr,
        [&](vstd::unique_ptr<DXByteBlob> const &buffer) {
            auto f = fopen(path.c_str(), "wb");
            if (f) {
                auto disp = vstd::create_disposer([&] { fclose(f); });
                auto serData = ShaderSerializer::Serialize(
                    str.properties,
                    {buffer->GetBufferPtr(), buffer->GetBufferSize()},
                    md5,
                    blockSize);
                fwrite(serData.data(), serData.size(), 1, f);
                //std::cout << "Save cache success!"sv << '\n';
            }
            auto cs = new ComputeShader(
                blockSize,
                str.properties,
                {buffer->GetBufferPtr(),
                 buffer->GetBufferSize()},
                device,
                md5);
            savePso(cs);
            return cs;
        },
        [](auto &&err) {
            std::cout << err << '\n';
            VSTL_ABORT();
            return nullptr;
        });
}
ComputeShader::ComputeShader(
    uint3 blockSize,
    vstd::span<std::pair<vstd::string, Property> const> properties,
    vstd::span<vbyte const> binData,
    Device *device,
    vstd::Guid guid)
    : Shader(std::move(properties), device->device.Get()),
      blockSize(blockSize),
      device(device),
      guid(guid) {
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
    vstd::span<std::pair<vstd::string, Property> const> prop,
    vstd::Guid guid,
    ComPtr<ID3D12RootSignature> &&rootSig,
    ComPtr<ID3D12PipelineState> &&pso)
    : device(device),
      blockSize(blockSize),
      Shader(prop, std::move(rootSig)),
      guid(guid),
      pso(std::move(pso)) {
}

ComputeShader::~ComputeShader() {
}
}// namespace toolhub::directx