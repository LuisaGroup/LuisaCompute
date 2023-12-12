#include <Shader/ShaderSerializer.h>
#include <Shader/ComputeShader.h>
#include <Shader/RasterShader.h>
#include <DXRuntime/GlobalSamplers.h>
#include <luisa/core/logging.h>
namespace lc::dx {
namespace shader_ser {
struct Header {
    uint64 headerVersion;
    vstd::MD5 md5;
    vstd::MD5 typeMD5;
    uint64 rootSigBytes;
    uint64 codeBytes;
    uint blockSize[3];
    uint propertyCount;
    uint bindlessCount;
    uint kernelArgCount;
    uint printerCount;
};
struct RasterHeader {
    uint64 headerVersion;
    vstd::MD5 md5;
    vstd::MD5 typeMD5;
    uint64 rootSigBytes;
    uint64 vertCodeBytes;
    uint64 pixelCodeBytes;
    uint propertyCount;
    uint bindlessCount;
    uint kernelArgCount;
    uint printerCount;
};
}// namespace shader_ser
namespace detail {
void SerPrinterSize(std::pair<vstd::string, Type const *> const &printer, vstd::vector<std::byte> &vec) {
    std::pair<size_t, size_t> strAndTypeSize{printer.first.size(), printer.second->description().size()};
    auto lastSize = vec.size();
    vec.push_back_uninitialized(strAndTypeSize.first + strAndTypeSize.second + sizeof(strAndTypeSize));
    auto ptr = vec.data() + lastSize;
    memcpy(ptr, &strAndTypeSize, sizeof(strAndTypeSize));
    ptr += sizeof(strAndTypeSize);
    memcpy(ptr, printer.first.data(), strAndTypeSize.first);
    ptr += strAndTypeSize.first;
    memcpy(ptr, printer.second->description().data(), strAndTypeSize.second);
}
std::pair<vstd::string, Type const *> DeserPrinterSize(BinaryStream *streamer) {
    std::pair<size_t, size_t> strAndTypeSize;
    streamer->read({reinterpret_cast<std::byte *>(&strAndTypeSize), sizeof(strAndTypeSize)});
    std::pair<vstd::string, Type const *> r;
    r.first.resize(strAndTypeSize.first);
    vstd::vector<char> typeDesc;
    typeDesc.resize_uninitialized(strAndTypeSize.second);
    streamer->read({reinterpret_cast<std::byte *>(r.first.data()), strAndTypeSize.first});
    streamer->read({reinterpret_cast<std::byte *>(typeDesc.data()), strAndTypeSize.second});
    r.second = Type::from(vstd::string_view{typeDesc.data(), typeDesc.size()});
    return r;
}
}// namespace detail
static constexpr size_t kRootSigReserveSize = 16384;
static constexpr uint64 kHeaderVersion = 1ull;
vstd::vector<std::byte>
ShaderSerializer::Serialize(
    vstd::span<hlsl::Property const> properties,
    vstd::span<SavedArgument const> kernelArgs,
    vstd::span<std::byte const> binByte,
    vstd::MD5 const &checkMD5,
    vstd::MD5 const &typeMD5,
    uint bindlessCount,
    uint3 blockSize,
    vstd::span<std::pair<vstd::string, Type const *> const> printers) {
    using namespace shader_ser;
    vstd::vector<std::byte> result;
    result.reserve(sizeof(Header) + binByte.size_bytes() + properties.size_bytes() + kernelArgs.size_bytes() + kRootSigReserveSize);
    result.push_back_uninitialized(sizeof(Header));
    for (auto &i : printers) {
        detail::SerPrinterSize(i, result);
    }
    Header header = {
        .headerVersion = kHeaderVersion,
        .md5 = checkMD5,
        .typeMD5 = typeMD5,
        .rootSigBytes = (uint64)SerializeRootSig(properties, result, false),
        .codeBytes = (uint64)binByte.size(),
        .propertyCount = static_cast<uint>(properties.size()),
        .bindlessCount = bindlessCount,
        .kernelArgCount = static_cast<uint>(kernelArgs.size()),
        .printerCount = static_cast<uint>(printers.size())};
    for (auto i : vstd::range(3)) {
        header.blockSize[i] = blockSize[i];
    }
    *reinterpret_cast<Header *>(result.data()) = header;
    vstd::push_back_all(result, binByte);
    vstd::push_back_all(result,
                        reinterpret_cast<std::byte const *>(properties.data()),
                        properties.size_bytes());
    vstd::push_back_all(result,
                        reinterpret_cast<std::byte const *>(kernelArgs.data()),
                        kernelArgs.size_bytes());
    return result;
}
vstd::vector<std::byte> ShaderSerializer::RasterSerialize(
    vstd::span<hlsl::Property const> properties,
    vstd::span<SavedArgument const> kernelArgs,
    vstd::span<std::byte const> vertBin,
    vstd::span<std::byte const> pixelBin,
    vstd::MD5 const &checkMD5,
    vstd::MD5 const &typeMD5,
    uint bindlessCount,
    vstd::span<std::pair<vstd::string, Type const *> const> printers) {
    using namespace shader_ser;
    vstd::vector<std::byte> result;
    result.reserve(sizeof(RasterHeader) + vertBin.size_bytes() + pixelBin.size_bytes() + properties.size_bytes() + kernelArgs.size_bytes() + kRootSigReserveSize);
    result.push_back_uninitialized(sizeof(RasterHeader));
    RasterHeader header = {
        .headerVersion = kHeaderVersion,
        .md5 = checkMD5,
        .typeMD5 = typeMD5,
        .rootSigBytes = (uint64)SerializeRootSig(properties, result, true),
        .vertCodeBytes = (uint64)vertBin.size(),
        .pixelCodeBytes = (uint64)pixelBin.size(),
        .propertyCount = static_cast<uint>(properties.size()),
        .bindlessCount = bindlessCount,
        .kernelArgCount = static_cast<uint>(kernelArgs.size()),
        .printerCount = static_cast<uint>(printers.size())};
    *reinterpret_cast<RasterHeader *>(result.data()) = std::move(header);
    for (auto &i : printers) {
        detail::SerPrinterSize(i, result);
    }
    vstd::push_back_all(result, vertBin);
    vstd::push_back_all(result, pixelBin);
    vstd::push_back_all(result,
                        reinterpret_cast<std::byte const *>(properties.data()),
                        properties.size_bytes());
    vstd::push_back_all(result,
                        reinterpret_cast<std::byte const *>(kernelArgs.data()),
                        kernelArgs.size_bytes());
    return result;
}
bool ShaderSerializer::CheckMD5(
    vstd::string_view fileName,
    vstd::MD5 const &checkMD5,
    luisa::BinaryIO const &streamFunc) {
    using namespace shader_ser;
    auto binStream = streamFunc.read_shader_bytecode(fileName);
    if (binStream == nullptr) return false;
    std::pair<uint64, vstd::MD5> versionAndMD5;
    binStream->read({reinterpret_cast<std::byte *>(&versionAndMD5),
                     sizeof(versionAndMD5)});

    return versionAndMD5.first == kHeaderVersion && versionAndMD5.second == checkMD5;
}
ComputeShader *ShaderSerializer::DeSerialize(
    vstd::string_view name,
    luisa::string_view psoName,
    CacheType cacheType,
    Device *device,
    luisa::BinaryIO const &streamFunc,
    vstd::optional<vstd::MD5> const &checkMD5,
    vstd::MD5 &typeMD5,
    vstd::vector<luisa::compute::Argument> &&bindings,
    bool &clearCache) {
    using namespace shader_ser;
    auto binStream = ReadBinaryIO(cacheType, &streamFunc, name);
    if (binStream == nullptr || binStream->length() <= sizeof(Header)) return nullptr;
    Header header;
    binStream->read({reinterpret_cast<std::byte *>(&header),
                     sizeof(Header)});
    if (header.headerVersion != kHeaderVersion || (checkMD5 && header.md5 != *checkMD5)) return nullptr;
    // TODO: printer
    vstd::vector<std::pair<vstd::string, Type const *>> printers;
    vstd::push_back_func(
        printers,
        header.printerCount,
        [&](size_t i) {
            return detail::DeserPrinterSize(binStream.get());
        });
    size_t targetSize =
        header.rootSigBytes +
        header.codeBytes +
        header.propertyCount * sizeof(hlsl::Property) +
        header.kernelArgCount * sizeof(SavedArgument);
    typeMD5 = header.typeMD5;
    vstd::vector<std::byte> binCode;
    binCode.push_back_uninitialized(targetSize);
    vstd::vector<std::byte> psoCode;

    binStream->read({binCode.data(), binCode.size()});
    auto psoStream = streamFunc.read_shader_cache(psoName);
    if (psoStream != nullptr && psoStream->length() > 0) {
        psoCode.push_back_uninitialized(psoStream->length());
        psoStream->read({psoCode.data(), psoCode.size()});
    }
    auto binPtr = binCode.data();
    auto rootSig = DeSerializeRootSig(
        device->device,
        {reinterpret_cast<std::byte const *>(binPtr), header.rootSigBytes});
    binPtr += header.rootSigBytes;
    // Try pipeline library
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc;
    psoDesc.NodeMask = 0;
    psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    psoDesc.pRootSignature = rootSig.Get();
    ComPtr<ID3D12PipelineState> pso;
    psoDesc.CS.pShaderBytecode = binPtr;
    psoDesc.CS.BytecodeLength = header.codeBytes;
    binPtr += header.codeBytes;
    psoDesc.CachedPSO.CachedBlobSizeInBytes = psoCode.size();
    auto createPipe = [&] {
        return device->device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(pso.GetAddressOf()));
    };
    // use PSO cache
    if (psoCode.empty()) {
        // No PSO
        clearCache = true;
        psoDesc.CachedPSO.pCachedBlob = nullptr;
        ThrowIfFailed(createPipe());
    } else {
        psoDesc.CachedPSO.pCachedBlob = psoCode.data();
        auto psoGenSuccess = createPipe();
        if (psoGenSuccess != S_OK) {
            // PSO cache miss(probably driver's version or hardware transformed), discard cache
            clearCache = true;
            LUISA_VERBOSE("{} pipeline cache illegal, discarded.", name);
            if (pso == nullptr) {
                psoDesc.CachedPSO.CachedBlobSizeInBytes = 0;
                psoDesc.CachedPSO.pCachedBlob = nullptr;
                ThrowIfFailed(createPipe());
            }
        }
    }
    vstd::vector<hlsl::Property> properties;
    vstd::vector<SavedArgument> kernelArgs;
    properties.push_back_uninitialized(header.propertyCount);
    kernelArgs.push_back_uninitialized(header.kernelArgCount);
    memcpy(properties.data(), binPtr, properties.size_bytes());
    binPtr += properties.size_bytes();
    memcpy(kernelArgs.data(), binPtr, kernelArgs.size_bytes());

    auto cs = new ComputeShader(
        uint3(header.blockSize[0], header.blockSize[1], header.blockSize[2]),
        device,
        std::move(properties),
        std::move(kernelArgs),
        std::move(bindings),
        std::move(printers),
        std::move(rootSig),
        std::move(pso));
    cs->bindlessCount = header.bindlessCount;
    return cs;
}
RasterShader *ShaderSerializer::RasterDeSerialize(
    luisa::string_view name,
    CacheType cacheType,
    Device *device,
    luisa::BinaryIO const &streamFunc,
    vstd::optional<vstd::MD5> const &ilMd5,
    vstd::MD5 &typeMD5,
    MeshFormat const &meshFormat) {
    using namespace shader_ser;
    auto binStream = ReadBinaryIO(cacheType, &streamFunc, name);
    if (binStream == nullptr || binStream->length() <= sizeof(RasterHeader)) return nullptr;
    RasterHeader header;
    binStream->read(
        {reinterpret_cast<std::byte *>(&header),
         sizeof(RasterHeader)});
    if (header.headerVersion != kHeaderVersion || (ilMd5 && header.md5 != *ilMd5)) return nullptr;
    // TODO: printer
    vstd::vector<std::pair<vstd::string, Type const *>> printers;
    vstd::push_back_func(
        printers,
        header.printerCount,
        [&](size_t i) {
            return detail::DeserPrinterSize(binStream.get());
        });
    size_t targetSize =
        header.rootSigBytes +
        header.vertCodeBytes +
        header.pixelCodeBytes +
        header.propertyCount * sizeof(hlsl::Property) +
        header.kernelArgCount * sizeof(SavedArgument);
    typeMD5 = header.typeMD5;
    vstd::vector<std::byte> binCode;
    binCode.push_back_uninitialized(targetSize);
    binStream->read({binCode.data(), binCode.size()});
    auto binPtr = binCode.data();
    // auto psoDesc = RasterShader::GetState(
    //     elements,
    //     meshFormat,
    //     state,
    //     rtv,
    //     dsv);

    // vstd::vector<D3D12_INPUT_ELEMENT_DESC> elements;
    // auto psoStream = streamFunc.read_shader_cache(psoName);
    // if (psoStream != nullptr && psoStream->length() > 0) {
    //     psoCode.push_back_uninitialized(psoStream->length());
    //     psoStream->read({psoCode.data(), psoCode.size()});
    //     psoDesc.CachedPSO = {
    //         .pCachedBlob = psoCode.data(),
    //         .CachedBlobSizeInBytes = psoCode.size()};
    // }
    auto rootSig = DeSerializeRootSig(
        device->device,
        {reinterpret_cast<std::byte const *>(binPtr), header.rootSigBytes});
    binPtr += header.rootSigBytes;
    // psoDesc.pRootSignature = rootSig.Get();
    vstd::vector<std::byte> vertBin;
    vertBin.push_back_uninitialized(header.vertCodeBytes);
    memcpy(vertBin.data(), binPtr, header.vertCodeBytes);
    binPtr += header.vertCodeBytes;
    vstd::vector<std::byte> pixelBin;
    pixelBin.push_back_uninitialized(header.pixelCodeBytes);
    memcpy(pixelBin.data(), binPtr, header.pixelCodeBytes);
    binPtr += header.pixelCodeBytes;
    // ComPtr<ID3D12PipelineState> pso;
    // auto createPipe = [&] {
    //     return device->device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(pso.GetAddressOf()));
    // };

    // use PSO cache
    // if (psoCode.empty()) {
    //     // No PSO
    //     clearCache = true;
    //     psoDesc.CachedPSO.pCachedBlob = nullptr;
    //     ThrowIfFailed(createPipe());
    // } else {
    //     auto psoGenSuccess = createPipe();
    //     if (psoGenSuccess != S_OK) {
    //         // PSO cache miss(probably driver's version or hardware transformed), discard cache
    //         clearCache = true;
    //         LUISA_VERBOSE("{} pipeline cache illegal, discarded.", name);
    //         if (pso == nullptr) {
    //             psoDesc.CachedPSO.CachedBlobSizeInBytes = 0;
    //             psoDesc.CachedPSO.pCachedBlob = nullptr;
    //             ThrowIfFailed(createPipe());
    //         }
    //     }
    // }
    vstd::vector<hlsl::Property> properties;
    vstd::vector<SavedArgument> kernelArgs;
    properties.push_back_uninitialized(header.propertyCount);
    kernelArgs.push_back_uninitialized(header.kernelArgCount);
    memcpy(properties.data(), binPtr, properties.size_bytes());
    binPtr += properties.size_bytes();
    memcpy(kernelArgs.data(), binPtr, kernelArgs.size_bytes());
    auto s = new RasterShader(
        device,
        header.md5,
        meshFormat,
        std::move(properties),
        std::move(kernelArgs),
        std::move(rootSig),
        std::move(printers),
        std::move(vertBin),
        std::move(pixelBin));
    s->bindlessCount = header.bindlessCount;
    return s;
}
ComPtr<ID3DBlob> ShaderSerializer::SerializeRootSig(
    vstd::span<hlsl::Property const> properties, bool isRasterShader) {
    vstd::fixed_vector<CD3DX12_ROOT_PARAMETER, 32> allParameter;
    allParameter.reserve(properties.size() + (isRasterShader ? 1 : 0));
    vstd::fixed_vector<CD3DX12_DESCRIPTOR_RANGE, 32> allRange;
    for (auto &&var : properties) {
        switch (var.type) {
            case hlsl::ShaderVariableType::UAVBufferHeap:
            case hlsl::ShaderVariableType::UAVTextureHeap:
            case hlsl::ShaderVariableType::CBVBufferHeap:
            case hlsl::ShaderVariableType::SamplerHeap:
            case hlsl::ShaderVariableType::SRVBufferHeap:
            case hlsl::ShaderVariableType::SRVTextureHeap: {
                allRange.emplace_back();
            } break;
            default:
                break;
        }
    }
    size_t offset = 0;
    for (auto &&var : properties) {

        switch (var.type) {
            case hlsl::ShaderVariableType::SRVTextureHeap:
            case hlsl::ShaderVariableType::SRVBufferHeap: {
                CD3DX12_DESCRIPTOR_RANGE &range = allRange[offset];
                offset++;
                range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, var.array_size, var.register_index, var.space_index);
                allParameter.emplace_back().InitAsDescriptorTable(1, &range);
            } break;
            case hlsl::ShaderVariableType::CBVBufferHeap: {
                CD3DX12_DESCRIPTOR_RANGE &range = allRange[offset];
                offset++;
                range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, var.array_size, var.register_index, var.space_index);
                allParameter.emplace_back().InitAsDescriptorTable(1, &range);
            } break;
            case hlsl::ShaderVariableType::SamplerHeap: {
                CD3DX12_DESCRIPTOR_RANGE &range = allRange[offset];
                offset++;
                range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, var.array_size, var.register_index, var.space_index);
                allParameter.emplace_back().InitAsDescriptorTable(1, &range);
            } break;
            case hlsl::ShaderVariableType::UAVTextureHeap:
            case hlsl::ShaderVariableType::UAVBufferHeap: {
                CD3DX12_DESCRIPTOR_RANGE &range = allRange[offset];
                offset++;
                range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, var.array_size, var.register_index, var.space_index);
                allParameter.emplace_back().InitAsDescriptorTable(1, &range);
            } break;
            case hlsl::ShaderVariableType::ConstantBuffer:
                allParameter.emplace_back().InitAsConstantBufferView(var.register_index, var.space_index);
                break;
            case hlsl::ShaderVariableType::ConstantValue:
                allParameter.emplace_back().InitAsConstants(var.space_index, var.register_index);
                break;
            case hlsl::ShaderVariableType::StructuredBuffer:
                allParameter.emplace_back().InitAsShaderResourceView(var.register_index, var.space_index);
                break;
            case hlsl::ShaderVariableType::RWStructuredBuffer:
                allParameter.emplace_back().InitAsUnorderedAccessView(var.register_index, var.space_index);
                break;
            default: assert(false); break;
        }
    }
    if (isRasterShader) {
        allParameter.emplace_back().InitAsConstants(1, 0);
    }
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc(
        allParameter.size(), allParameter.data(),
        0, nullptr,
        isRasterShader ? D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT : D3D12_ROOT_SIGNATURE_FLAG_NONE);
    Microsoft::WRL::ComPtr<ID3DBlob> serializedRootSig;
    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;
    ThrowIfFailed(D3D12SerializeVersionedRootSignature(
        &rootSigDesc,
        serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf()));
    if (errorBlob && errorBlob->GetBufferSize() > 0) [[unlikely]] {
        LUISA_ERROR("Serialize root signature error: {}", vstd::string_view(reinterpret_cast<char const *>(errorBlob->GetBufferPointer()), errorBlob->GetBufferSize()));
    }
    return serializedRootSig;
}
size_t ShaderSerializer::SerializeRootSig(
    vstd::span<hlsl::Property const> properties,
    vstd::vector<std::byte> &result,
    bool isRasterShader) {
    auto lastSize = result.size();
    auto blob = SerializeRootSig(properties, isRasterShader);
    vstd::push_back_all(
        result,
        (std::byte const *)blob->GetBufferPointer(),
        blob->GetBufferSize());
    return result.size() - lastSize;
}
ComPtr<ID3D12RootSignature> ShaderSerializer::DeSerializeRootSig(
    ID3D12Device *device,
    vstd::span<std::byte const> bytes) {
    ComPtr<ID3D12RootSignature> rootSig;
    ThrowIfFailed(device->CreateRootSignature(
        0,
        bytes.data(),
        bytes.size(),
        IID_PPV_ARGS(rootSig.GetAddressOf())));
    return rootSig;
}
vstd::vector<SavedArgument> ShaderSerializer::SerializeKernel(Function kernel) {
    assert(kernel.tag() != Function::Tag::CALLABLE);
    auto &&args = kernel.arguments();
    vstd::vector<SavedArgument> result;
    vstd::push_back_func(result, args.size(), [&](size_t i) {
        auto &&var = args[i];
        return SavedArgument(kernel, var);
    });
    return result;
}

vstd::vector<SavedArgument> ShaderSerializer::SerializeKernel(
    vstd::IRange<std::pair<Variable, Usage>> &arguments) {
    vstd::vector<SavedArgument> result;
    for (auto &&i : arguments) {
        result.emplace_back(i.second, i.first);
    }
    return result;
}
}// namespace lc::dx
