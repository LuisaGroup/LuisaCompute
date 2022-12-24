
#include <Shader/ShaderSerializer.h>
#include <Shader/ComputeShader.h>
#include <Shader/RasterShader.h>
#include <DXRuntime/GlobalSamplers.h>
#include <vstl/small_vector.h>
#include <core/logging.h>
namespace toolhub::directx {
namespace shader_ser {
struct Header {
    vstd::MD5 md5;
    vstd::MD5 typeMD5;
    uint64 rootSigBytes;
    uint64 codeBytes;
    uint blockSize[3];
    uint propertyCount;
    uint bindlessCount;
    uint kernelArgCount;
};
struct RasterHeader {
    vstd::MD5 md5;
    vstd::MD5 typeMD5;
    uint64 rootSigBytes;
    uint64 vertCodeBytes;
    uint64 pixelCodeBytes;
    uint propertyCount;
    uint bindlessCount;
    uint kernelArgCount;
};
}// namespace shader_ser
static constexpr size_t kRootSigReserveSize = 16384;
vstd::vector<std::byte>
ShaderSerializer::Serialize(
    vstd::span<Property const> properties,
    vstd::span<SavedArgument const> kernelArgs,
    vstd::span<std::byte const> binByte,
    vstd::MD5 const &checkMD5,
    vstd::MD5 const &typeMD5,
    uint bindlessCount,
    uint3 blockSize) {
    using namespace shader_ser;
    vstd::vector<std::byte> result;
    result.reserve(sizeof(Header) + binByte.size_bytes() + properties.size_bytes() + kernelArgs.size_bytes() + kRootSigReserveSize);
    result.push_back_uninitialized(sizeof(Header));
    Header header = {
        .md5 = checkMD5,
        .typeMD5 = typeMD5,
        .rootSigBytes = (uint64)SerializeRootSig(properties, result, false),
        .codeBytes = (uint64)binByte.size(),
        .propertyCount = static_cast<uint>(properties.size()),
        .bindlessCount = bindlessCount,
        .kernelArgCount = static_cast<uint>(kernelArgs.size())};
    for (auto i : vstd::range(3)) {
        header.blockSize[i] = blockSize[i];
    }
    *reinterpret_cast<Header *>(result.data()) = std::move(header);
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
    vstd::span<Property const> properties,
    vstd::span<SavedArgument const> kernelArgs,
    vstd::span<std::byte const> vertBin,
    vstd::span<std::byte const> pixelBin,
    vstd::MD5 const &checkMD5,
    vstd::MD5 const &typeMD5,
    uint bindlessCount) {
    using namespace shader_ser;
    vstd::vector<std::byte> result;
    result.reserve(sizeof(RasterHeader) + vertBin.size_bytes() + pixelBin.size_bytes() + properties.size_bytes() + kernelArgs.size_bytes() + kRootSigReserveSize);
    result.push_back_uninitialized(sizeof(RasterHeader));
    RasterHeader header = {
        .md5 = checkMD5,
        .typeMD5 = typeMD5,
        .rootSigBytes = (uint64)SerializeRootSig(properties, result, true),
        .vertCodeBytes = (uint64)vertBin.size(),
        .pixelCodeBytes = (uint64)pixelBin.size(),
        .propertyCount = static_cast<uint>(properties.size()),
        .bindlessCount = bindlessCount,
        .kernelArgCount = static_cast<uint>(kernelArgs.size())};
    *reinterpret_cast<RasterHeader *>(result.data()) = std::move(header);
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
    BinaryIO &streamFunc) {
    using namespace shader_ser;
    auto binStream = streamFunc.read_bytecode(fileName);
    if (binStream == nullptr) return false;
    vstd::MD5 md5;
    binStream->read({reinterpret_cast<std::byte *>(&md5),
                     sizeof(vstd::MD5)});

    return md5 == checkMD5;
}
ComputeShader *ShaderSerializer::DeSerialize(
    vstd::string_view name,
    luisa::string_view psoName,
    FileType fileType,
    Device *device,
    BinaryIO &streamFunc,
    vstd::optional<vstd::MD5> const &checkMD5,
    vstd::MD5 &typeMD5,
    bool &clearCache) {
    using namespace shader_ser;

    auto binStream = [&] {
        switch (fileType) {
            case FileType::ByteCode:
                return streamFunc.read_bytecode(name);
            case FileType::Cache:
                return streamFunc.read_cache(name);
            case FileType::Internal:
                return streamFunc.read_internal(name);
        }
    }();
    if (binStream == nullptr || binStream->length() <= sizeof(Header)) return nullptr;
    Header header;
    binStream->read({reinterpret_cast<std::byte *>(&header),
                     sizeof(Header)});
    if (checkMD5 && header.md5 != *checkMD5) return nullptr;
    size_t targetSize =
        header.rootSigBytes +
        header.codeBytes +
        header.propertyCount * sizeof(Property) +
        header.kernelArgCount * sizeof(SavedArgument);
    typeMD5 = header.typeMD5;
    if (binStream->length() != sizeof(Header) + targetSize) {
        return nullptr;
    }
    vstd::vector<std::byte> binCode(targetSize);
    vstd::vector<std::byte> psoCode;

    binStream->read({binCode.data(), binCode.size()});
    auto psoStream = streamFunc.read_cache(psoName);
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
            LUISA_INFO("{} pipeline cache illegal, discarded.", name);
            if (pso == nullptr) {
                psoDesc.CachedPSO.CachedBlobSizeInBytes = 0;
                psoDesc.CachedPSO.pCachedBlob = nullptr;
                ThrowIfFailed(createPipe());
            }
        }
    }
    vstd::vector<Property> properties;
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
        std::move(rootSig),
        std::move(pso));
    cs->bindlessCount = header.bindlessCount;
    return cs;
}
RasterShader *ShaderSerializer::RasterDeSerialize(
    luisa::string_view name,
    luisa::string_view psoName,
    FileType fileType,
    Device *device,
    BinaryIO &streamFunc,
    vstd::optional<vstd::MD5> const &ilMd5,
    vstd::optional<vstd::MD5> &psoMd5,
    vstd::MD5 &typeMD5,
    MeshFormat const &meshFormat,
    RasterState const &state,
    vstd::span<PixelFormat const> rtv,
    DepthFormat dsv,
    bool &clearCache) {
    using namespace shader_ser;
    auto binStream = [&] {
        switch (fileType) {
            case FileType::ByteCode:
                return streamFunc.read_bytecode(name);
            case FileType::Cache:
                return streamFunc.read_cache(name);
            case FileType::Internal:
                return streamFunc.read_internal(name);
        }
    }();
    if (binStream == nullptr || binStream->length() <= sizeof(RasterHeader)) return nullptr;
    RasterHeader header;
    binStream->read(
        {reinterpret_cast<std::byte *>(&header),
         sizeof(RasterHeader)});
    if (ilMd5 && header.md5 != *ilMd5) return nullptr;

    size_t targetSize =
        header.rootSigBytes +
        header.vertCodeBytes +
        header.pixelCodeBytes +
        header.propertyCount * sizeof(Property) +
        header.kernelArgCount * sizeof(SavedArgument);
    typeMD5 = header.typeMD5;
    if (binStream->length() != targetSize + sizeof(RasterHeader)) {
        return nullptr;
    }
    vstd::vector<std::byte> binCode;
    vstd::vector<std::byte> psoCode;
    binCode.push_back_uninitialized(targetSize);
    binStream->read({binCode.data(), binCode.size()});
    auto binPtr = binCode.data();
    vstd::vector<D3D12_INPUT_ELEMENT_DESC> elements;
    auto psoDesc = RasterShader::GetState(
        elements,
        meshFormat,
        state,
        rtv,
        dsv);

    auto psoStream = streamFunc.read_cache(psoName);
    if (psoStream != nullptr && psoStream->length() > 0) {
        psoCode.push_back_uninitialized(psoStream->length());
        psoStream->read({psoCode.data(), psoCode.size()});
        psoDesc.CachedPSO = {
            .pCachedBlob = psoCode.data(),
            .CachedBlobSizeInBytes = psoCode.size()};
    }
    auto rootSig = DeSerializeRootSig(
        device->device,
        {reinterpret_cast<std::byte const *>(binPtr), header.rootSigBytes});
    binPtr += header.rootSigBytes;
    psoDesc.pRootSignature = rootSig.Get();
    psoDesc.VS = {
        .pShaderBytecode = binPtr,
        .BytecodeLength = header.vertCodeBytes};
    binPtr += header.vertCodeBytes;
    psoDesc.PS = {
        .pShaderBytecode = binPtr,
        .BytecodeLength = header.pixelCodeBytes};
    binPtr += header.pixelCodeBytes;
    ComPtr<ID3D12PipelineState> pso;
    auto createPipe = [&] {
        return device->device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(pso.GetAddressOf()));
    };

    // use PSO cache
    if (psoCode.empty()) {
        // No PSO
        clearCache = true;
        psoDesc.CachedPSO.pCachedBlob = nullptr;
        ThrowIfFailed(createPipe());
    } else {
        auto psoGenSuccess = createPipe();
        if (psoGenSuccess != S_OK) {
            // PSO cache miss(probably driver's version or hardware transformed), discard cache
            clearCache = true;
            LUISA_INFO("{} pipeline cache illegal, discarded.", name);
            if (pso == nullptr) {
                psoDesc.CachedPSO.CachedBlobSizeInBytes = 0;
                psoDesc.CachedPSO.pCachedBlob = nullptr;
                ThrowIfFailed(createPipe());
            }
        }
    }
    vstd::vector<Property> properties;
    vstd::vector<SavedArgument> kernelArgs;
    properties.push_back_uninitialized(header.propertyCount);
    kernelArgs.push_back_uninitialized(header.kernelArgCount);
    memcpy(properties.data(), binPtr, properties.size_bytes());
    binPtr += properties.size_bytes();
    memcpy(kernelArgs.data(), binPtr, kernelArgs.size_bytes());
    auto s = new RasterShader(
        device,
        std::move(properties),
        std::move(kernelArgs),
        std::move(rootSig),
        std::move(pso),
        state.topology);
    s->bindlessCount = header.bindlessCount;
    return s;
}
ComPtr<ID3DBlob> ShaderSerializer::SerializeRootSig(
    vstd::span<Property const> properties, bool isRasterShader) {
    vstd::fixed_vector<CD3DX12_ROOT_PARAMETER, 32> allParameter;
    allParameter.reserve(properties.size() + (isRasterShader ? 1 : 0));
    vstd::fixed_vector<CD3DX12_DESCRIPTOR_RANGE, 32> allRange;
    for (auto &&var : properties) {
        switch (var.type) {
            case ShaderVariableType::UAVDescriptorHeap:
            case ShaderVariableType::CBVDescriptorHeap:
            case ShaderVariableType::SampDescriptorHeap:
            case ShaderVariableType::SRVDescriptorHeap: {
                allRange.emplace_back();
            } break;
            default:
                break;
        }
    }
    size_t offset = 0;
    for (auto &&var : properties) {

        switch (var.type) {
            case ShaderVariableType::SRVDescriptorHeap: {
                CD3DX12_DESCRIPTOR_RANGE &range = allRange[offset];
                offset++;
                range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, var.arrSize == 0 ? 1 : var.arrSize, var.registerIndex, var.spaceIndex);
                allParameter.emplace_back().InitAsDescriptorTable(1, &range);
            } break;
            case ShaderVariableType::CBVDescriptorHeap: {
                CD3DX12_DESCRIPTOR_RANGE &range = allRange[offset];
                offset++;
                range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, var.arrSize == 0 ? 1 : var.arrSize, var.registerIndex, var.spaceIndex);
                allParameter.emplace_back().InitAsDescriptorTable(1, &range);
            } break;
            case ShaderVariableType::SampDescriptorHeap: {
                CD3DX12_DESCRIPTOR_RANGE &range = allRange[offset];
                offset++;
                range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, var.arrSize == 0 ? 1 : var.arrSize, var.registerIndex, var.spaceIndex);
                allParameter.emplace_back().InitAsDescriptorTable(1, &range);
            } break;
            case ShaderVariableType::UAVDescriptorHeap: {
                CD3DX12_DESCRIPTOR_RANGE &range = allRange[offset];
                offset++;
                range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, var.arrSize == 0 ? 1 : var.arrSize, var.registerIndex, var.spaceIndex);
                allParameter.emplace_back().InitAsDescriptorTable(1, &range);
            } break;
            case ShaderVariableType::ConstantBuffer:
                allParameter.emplace_back().InitAsConstantBufferView(var.registerIndex, var.spaceIndex);
                break;
            case ShaderVariableType::ConstantValue:
                allParameter.emplace_back().InitAsConstants(var.spaceIndex, var.registerIndex);
                break;
            case ShaderVariableType::StructuredBuffer:
                allParameter.emplace_back().InitAsShaderResourceView(var.registerIndex, var.spaceIndex);
                break;
            case ShaderVariableType::RWStructuredBuffer:
                allParameter.emplace_back().InitAsUnorderedAccessView(var.registerIndex, var.spaceIndex);
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
    return serializedRootSig;
}
size_t ShaderSerializer::SerializeRootSig(
    vstd::span<Property const> properties,
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
    assert(kernel.tag() == Function::Tag::KERNEL);
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
}// namespace toolhub::directx