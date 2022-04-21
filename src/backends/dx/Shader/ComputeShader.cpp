
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
    CodegenResult const &str,
    uint3 blockSize,
    uint shaderModel,
    vstd::optional<vstd::string> &&cachePath) {
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
    if (cachePath) {
        path = std::move(*cachePath);
    } else {
        path.reserve(64);
        path << ".cache/" << str.md5.ToString();
    }
    psoPath = path;
    psoPath << ".pso";
    static constexpr bool USE_CACHE = true;
    if constexpr (USE_CACHE) {
        SerializeVisitor visitor(
            path,
            psoPath);
        //Cached
        if (visitor.csoReader) {
            auto result = ShaderSerializer::DeSerialize(
                str.properties,
                device,
                str.md5,
                visitor);
            if (result) {
                result->bindlessCount = str.bdlsBufferCount;
                //std::cout << "Read cache success!"sv << '\n';
                if (visitor.oldDeleted) {
                    savePso(result);
                }
                return result;
            }
        }
    }

    auto compResult = [&] {
        if constexpr (!USE_CACHE) {
            std::cout
                << "\n===============================\n"
                << str.result
                << "\n===============================\n";
        }
        return Device::Compiler()->CompileCompute(
            str.result,
            true,
            shaderModel);
    }();
    return compResult.multi_visit_or(
        (ComputeShader *)nullptr,
        [&](vstd::unique_ptr<DXByteBlob> const &buffer) {
            auto f = fopen(path.c_str(), "wb");
            if (f) {
                if constexpr (USE_CACHE) {
                    auto disp = vstd::create_disposer([&] { fclose(f); });
                    auto serData = ShaderSerializer::Serialize(
                        str.properties,
                        {buffer->GetBufferPtr(), buffer->GetBufferSize()},
                        str.md5,
                        blockSize);
                    fwrite(serData.data(), serData.size(), 1, f);
                }
            }
            auto cs = new ComputeShader(
                blockSize,
                str.properties,
                {buffer->GetBufferPtr(),
                 buffer->GetBufferSize()},
                device,
                str.md5);
            cs->bindlessCount = str.bdlsBufferCount;
            if constexpr (USE_CACHE) {
                savePso(cs);
            }
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