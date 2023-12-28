#include <Shader/RasterShader.h>
#include <Resource/DepthBuffer.h>
#include <Shader/ShaderSerializer.h>
#include "../../common/hlsl/hlsl_codegen.h"
#include "../../common/hlsl/shader_compiler.h"
#include <luisa/vstl/md5.h>
#include <luisa/core/logging.h>
namespace lc::dx {
namespace RasterShaderDetail {
static constexpr bool PRINT_CODE = false;
static vstd::vector<SavedArgument> GetKernelArgs(Function vertexKernel, Function pixelKernel) {
    if (vertexKernel.builder() == nullptr || pixelKernel.builder() == nullptr) {
        return {};
    } else {
        auto vertSpan = vertexKernel.arguments();
        auto vertArgs =
            vstd::range_linker{
                vstd::make_ite_range(vertSpan.subspan(1)),
                vstd::transform_range{
                    [&](Variable const &var) {
                        return std::pair<Variable, Usage>{var, vertexKernel.variable_usage(var.uid())};
                    }}};
        auto pixelSpan = pixelKernel.arguments();
        auto pixelArgs =
        vstd::range_linker{
            vstd::make_ite_range(pixelSpan.subspan(1)),
            vstd::transform_range{
                [&](Variable const &var) {
                    return std::pair<Variable, Usage>{var, pixelKernel.variable_usage(var.uid())};
                }}};
        auto args = vstd::tuple_range(std::move(vertArgs), std::move(pixelArgs)).i_range();
        return ShaderSerializer::SerializeKernel(args);
    }
}
}// namespace RasterShaderDetail
RasterShader::RasterShader(
    Device *device,
    vstd::MD5 md5,
    MeshFormat const &meshFormat,
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ComPtr<ID3D12RootSignature> &&rootSig,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    vstd::vector<std::byte> &&vertBinData,
    vstd::vector<std::byte> &&pixelBinData)
    : Shader(std::move(prop), std::move(args), std::move(rootSig), std::move(printers)), device(device), md5{md5},
      vertBinData{std::move(vertBinData)}, pixelBinData{std::move(pixelBinData)} {
    GetMeshFormatState(elements, meshFormat);
}
void RasterShader::GetMeshFormatState(
    vstd::vector<D3D12_INPUT_ELEMENT_DESC> &inputLayout,
    MeshFormat const &meshFormat) {
    inputLayout.clear();
    inputLayout.reserve(meshFormat.vertex_attribute_count());
    static auto SemanticName = {
        "POSITION",
        "NORMAL",
        "TANGENT",
        "COLOR",
        "UV",
        "UV",
        "UV",
        "UV"};
    static auto SemanticIndex = {0u, 0u, 0u, 0u, 0u, 1u, 2u, 3u};
    vstd::fixed_vector<uint, 4> offsets(meshFormat.vertex_stream_count());
    memset(offsets.data(), 0, offsets.size_bytes());
    for (auto i : vstd::range(meshFormat.vertex_stream_count())) {
        auto vec = meshFormat.attributes(i);
        for (auto &&attr : vec) {
            size_t size;
            DXGI_FORMAT format;
            switch (attr.format) {
                case VertexElementFormat::XYZW8UNorm:
                    size = 4;
                    format = DXGI_FORMAT_R8G8B8A8_UNORM;
                    break;
                case VertexElementFormat::XY16UNorm:
                    size = 4;
                    format = DXGI_FORMAT_R16G16_UNORM;
                    break;
                case VertexElementFormat::XYZW16UNorm:
                    size = 8;
                    format = DXGI_FORMAT_R16G16B16A16_UNORM;
                    break;

                case VertexElementFormat::XY16Float:
                    size = 4;
                    format = DXGI_FORMAT_R16G16_FLOAT;
                    break;
                case VertexElementFormat::XYZW16Float:
                    size = 8;
                    format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                    break;
                case VertexElementFormat::X32Float:
                    size = 4;
                    format = DXGI_FORMAT_R32_FLOAT;
                    break;
                case VertexElementFormat::XY32Float:
                    size = 8;
                    format = DXGI_FORMAT_R32G32_FLOAT;
                    break;
                case VertexElementFormat::XYZ32Float:
                    size = 12;
                    format = DXGI_FORMAT_R32G32B32_FLOAT;
                    break;
                case VertexElementFormat::XYZW32Float:
                    size = 16;
                    format = DXGI_FORMAT_R32G32B32A32_FLOAT;
                    break;
            }
            auto &offset = offsets[i];
            inputLayout.emplace_back(D3D12_INPUT_ELEMENT_DESC{
                .SemanticName = SemanticName.begin()[static_cast<uint>(attr.type)],
                .SemanticIndex = SemanticIndex.begin()[static_cast<uint>(attr.type)],
                .Format = format,
                .InputSlot = static_cast<uint>(i),
                .AlignedByteOffset = offset,
                .InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA});
            offset += size;
        }
    }
}
RasterShader::RasterShader(
    Device *device,
    vstd::MD5 md5,
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    MeshFormat const &meshFormat,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    vstd::vector<std::byte> &&vertBinData,
    vstd::vector<std::byte> &&pixelBinData)
    : Shader(std::move(prop), std::move(args), device->device.Get(), std::move(printers), true),
      device(device), md5{md5},
      vertBinData{std::move(vertBinData)}, pixelBinData{std::move(pixelBinData)} {
    GetMeshFormatState(elements, meshFormat);
}
RasterShader::~RasterShader() {}
D3D12_GRAPHICS_PIPELINE_STATE_DESC RasterShader::GetState(
    vstd::span<D3D12_INPUT_ELEMENT_DESC const> inputLayout,
    RasterState const &state,
    vstd::span<GFXFormat const> rtv,
    DepthFormat dsv) {
    auto GetBlendState = [](BlendWeight op) {
        switch (op) {
            case BlendWeight::One:
                return D3D12_BLEND_ONE;
            case BlendWeight::PrimColor:
                return D3D12_BLEND_SRC_COLOR;
            case BlendWeight::ImgColor:
                return D3D12_BLEND_DEST_COLOR;
            case BlendWeight::PrimAlpha:
                return D3D12_BLEND_SRC_ALPHA;
            case BlendWeight::ImgAlpha:
                return D3D12_BLEND_DEST_ALPHA;
            case BlendWeight::OneMinusPrimColor:
                return D3D12_BLEND_INV_SRC_COLOR;
            case BlendWeight::OneMinusImgColor:
                return D3D12_BLEND_INV_DEST_COLOR;
            case BlendWeight::OneMinusPrimAlpha:
                return D3D12_BLEND_INV_SRC_ALPHA;
            case BlendWeight::OneMinusImgAlpha:
                return D3D12_BLEND_INV_DEST_ALPHA;
            default:
                return D3D12_BLEND_ZERO;
        }
    };
    auto ComparisonState = [](Comparison c) {
        switch (c) {
            case Comparison::Never:
                return D3D12_COMPARISON_FUNC_NEVER;
            case Comparison::Less:
                return D3D12_COMPARISON_FUNC_LESS;
            case Comparison::Equal:
                return D3D12_COMPARISON_FUNC_EQUAL;
            case Comparison::LessEqual:
                return D3D12_COMPARISON_FUNC_LESS_EQUAL;
            case Comparison::Greater:
                return D3D12_COMPARISON_FUNC_GREATER;
            case Comparison::NotEqual:
                return D3D12_COMPARISON_FUNC_NOT_EQUAL;
            case Comparison::GreaterEqual:
                return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
            default:
                return D3D12_COMPARISON_FUNC_ALWAYS;
        }
    };
    auto StencilOpState = [](StencilOp s) {
        switch (s) {
            case StencilOp::Keep:
                return D3D12_STENCIL_OP_KEEP;
            case StencilOp::Replace:
                return D3D12_STENCIL_OP_REPLACE;
            default:
                return D3D12_STENCIL_OP_ZERO;
        }
    };
    auto GetCullMode = [](CullMode c) {
        switch (c) {
            case CullMode::None:
                return D3D12_CULL_MODE_NONE;
            case CullMode::Front:
                return D3D12_CULL_MODE_FRONT;
            default:
                return D3D12_CULL_MODE_BACK;
        }
    };
    auto GetTopology = [](TopologyType t) {
        switch (t) {
            case TopologyType::Point:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
            case TopologyType::Line:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            default:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        }
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC result = {
        .SampleMask = ~0u,
        .PrimitiveTopologyType = GetTopology(state.topology),
        .NumRenderTargets = static_cast<uint>(rtv.size()),
        .SampleDesc = {.Count = 1, .Quality = 0}};

    if (state.blend_state.enable_blend) {
        D3D12_RENDER_TARGET_BLEND_DESC blend{
            .RenderTargetWriteMask = 15};
        auto &v = state.blend_state;
        blend.BlendEnable = true;
        blend.BlendOp = D3D12_BLEND_OP_ADD;
        blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        blend.SrcBlend = GetBlendState(v.prim_op);
        blend.DestBlend = GetBlendState(v.img_op);
        blend.SrcBlendAlpha = D3D12_BLEND_ZERO;
        blend.DestBlendAlpha = D3D12_BLEND_ONE;
        blend.LogicOp = D3D12_LOGIC_OP_NOOP;

        result.BlendState = {
            .AlphaToCoverageEnable = false,
            .IndependentBlendEnable = false,
            .RenderTarget = {blend}};
    } else {
        D3D12_RENDER_TARGET_BLEND_DESC blend{
            .RenderTargetWriteMask = 15};
        result.BlendState = {
            .AlphaToCoverageEnable = false,
            .IndependentBlendEnable = false,
            .RenderTarget = {blend}};
    }

    D3D12_DEPTH_STENCIL_DESC &depth = result.DepthStencilState;
    if (state.depth_state.enable_depth) {
        auto &v = state.depth_state;
        depth.DepthEnable = true;
        depth.DepthFunc = ComparisonState(v.comparison);
        depth.DepthWriteMask = v.write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    }
    if (state.stencil_state.enable_stencil) {
        auto &v = state.stencil_state;
        depth.StencilEnable = true;
        depth.StencilReadMask = v.read_mask;
        depth.StencilWriteMask = v.write_mask;
        auto SetFace = [&](StencilFaceOp const &face, D3D12_DEPTH_STENCILOP_DESC &r) {
            r.StencilFunc = ComparisonState(face.comparison);
            r.StencilFailOp = StencilOpState(face.stencil_fail_op);
            r.StencilDepthFailOp = StencilOpState(face.depth_fail_op);
            r.StencilPassOp = StencilOpState(face.pass_op);
        };
        SetFace(v.front_face_op, depth.FrontFace);
        SetFace(v.back_face_op, depth.BackFace);
    }
    result.RasterizerState = {
        .FillMode = state.fill_mode == FillMode::Solid ? D3D12_FILL_MODE_SOLID : D3D12_FILL_MODE_WIREFRAME,
        .CullMode = GetCullMode(state.cull_mode),
        .FrontCounterClockwise = state.front_counter_clockwise,
        .DepthClipEnable = state.depth_clip,
        .ConservativeRaster = state.conservative ? D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON : D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF};
    for (auto i : vstd::range(rtv.size())) {
        result.RTVFormats[i] = static_cast<DXGI_FORMAT>(rtv[i]);
    }
    result.DSVFormat = static_cast<DXGI_FORMAT>(DepthBuffer::GetDepthFormat(dsv));
    result.InputLayout = {.pInputElementDescs = inputLayout.data(), .NumElements = static_cast<uint>(inputLayout.size())};
    return result;
}
vstd::MD5 RasterShader::GenMD5(
    vstd::MD5 const &codeMD5,
    MeshFormat const &meshFormat) {
    vstd::fixed_vector<uint64, 8> streamHashes;
    vstd::push_back_func(
        streamHashes,
        meshFormat.vertex_stream_count(),
        [&](size_t i) {
            auto atr = meshFormat.attributes(i);
            return vstd_xxhash_gethash(atr.data(), atr.size_bytes());
        });
    struct Hashes {
        vstd::MD5 codeMd5;
        uint64 meshFormatMD5;
    };
    Hashes h{
        .codeMd5 = codeMD5,
        .meshFormatMD5 = vstd_xxhash_gethash(streamHashes.data(), streamHashes.size_bytes())};
    return vstd::MD5(
        vstd::span<uint8_t const>{
            reinterpret_cast<uint8_t const *>(&h),
            sizeof(Hashes)});
}
RasterShader *RasterShader::CompileRaster(
    BinaryIO const *fileIo,
    Device *device,
    Function vertexKernel,
    Function pixelKernel,
    vstd::function<hlsl::CodegenResult()> const &codegen,
    vstd::MD5 const &md5,
    uint shaderModel,
    MeshFormat const &meshFormat,
    vstd::string_view fileName,
    CacheType cacheType,
    bool enableUnsafeMath) {
    auto CompileNewCompute = [&](bool writeCache) -> RasterShader * {
        auto str = codegen();
        uint bdlsBufferCount = 0;
        if (str.useBufferBindless) bdlsBufferCount++;
        if (str.useTex2DBindless) bdlsBufferCount++;
        if (str.useTex3DBindless) bdlsBufferCount++;
        if constexpr (RasterShaderDetail::PRINT_CODE) {
            auto f = fopen("hlsl_output.hlsl", "ab");
            fwrite(str.result.data(), str.result.size(), 1, f);
            fclose(f);
        }
        auto compResult = Device::Compiler()->compile_raster(
            str.result.view(),
            true,
            shaderModel, enableUnsafeMath,
            false);
        if (compResult.vertex.is_type_of<vstd::string>()) [[unlikely]] {
            LUISA_ERROR("DXC compile vertex-shader error: {}", compResult.vertex.get<1>());
            return nullptr;
        }
        if (compResult.pixel.is_type_of<vstd::string>()) [[unlikely]] {
            LUISA_ERROR("DXC compile pixel-shader error: {}", compResult.pixel.get<1>());
            return nullptr;
        }
        auto kernelArgs = RasterShaderDetail::GetKernelArgs(vertexKernel, pixelKernel);
        auto GetVector = [&](hlsl::DxcByteBlob const &blob) {
            vstd::vector<std::byte> vec;
            vec.push_back_uninitialized(blob.size());
            memcpy(vec.data(), blob.data(), blob.size());
            return vec;
        };
        auto vertBin = GetVector(*compResult.vertex.get<0>());
        auto pixelBin = GetVector(*compResult.pixel.get<0>());

        if (writeCache) {
            auto serData = ShaderSerializer::RasterSerialize(
                str.properties,
                kernelArgs, vertBin, pixelBin, md5, str.typeMD5, bdlsBufferCount,
                str.printers);
            WriteBinaryIO(cacheType, fileIo, fileName, {reinterpret_cast<std::byte const *>(serData.data()), serData.size_bytes()});
        }

        auto s = new RasterShader(
            device,
            md5,
            std::move(str.properties),
            std::move(kernelArgs),
            meshFormat,
            std::move(str.printers),
            std::move(vertBin),
            std::move(pixelBin));
        s->bindlessCount = bdlsBufferCount;
        return s;
    };

    if (!fileName.empty()) {
        // auto psoName = Shader::PSOName(device, fileName);
        vstd::MD5 typeMD5;
        auto result = ShaderSerializer::RasterDeSerialize(
            fileName, cacheType, device, *fileIo, md5,
            typeMD5, meshFormat);
        if (result) {
            return result;
        }
        return CompileNewCompute(true);
    } else {
        return CompileNewCompute(false);
    }
}
void RasterShader::SaveRaster(
    BinaryIO const *fileIo,
    Device *device,
    hlsl::CodegenResult const &str,
    vstd::MD5 const &md5,
    vstd::string_view fileName,
    Function vertexKernel,
    Function pixelKernel,
    uint shaderModel,
    bool enableUnsafeMath) {
    if constexpr (RasterShaderDetail::PRINT_CODE) {
        auto f = fopen("hlsl_output.hlsl", "ab");
        fwrite(str.result.data(), str.result.size(), 1, f);
        fclose(f);
    }
    if (ShaderSerializer::CheckMD5(fileName, md5, *fileIo)) return;
    auto compResult = Device::Compiler()->compile_raster(
        str.result.view(),
        true,
        shaderModel,
        enableUnsafeMath,
        false);

    if (compResult.vertex.is_type_of<vstd::string>()) [[unlikely]] {
        LUISA_ERROR("DXC compile vertex-shader error: {}", compResult.vertex.get<1>());
        return;
    }
    if (compResult.pixel.is_type_of<vstd::string>()) [[unlikely]] {
        LUISA_ERROR("DXC compile pixel-shader error: {}", compResult.pixel.get<1>());
        return;
    }
    auto kernelArgs = RasterShaderDetail::GetKernelArgs(vertexKernel, pixelKernel);
    auto GetSpan = [&](hlsl::DxcByteBlob const &blob) {
        return vstd::span<std::byte const>{blob.data(), blob.size()};
    };
    auto vertBin = GetSpan(*compResult.vertex.get<0>());
    auto pixelBin = GetSpan(*compResult.pixel.get<0>());
    uint bdlsBufferCount = 0;
    if (str.useBufferBindless) bdlsBufferCount++;
    if (str.useTex2DBindless) bdlsBufferCount++;
    if (str.useTex3DBindless) bdlsBufferCount++;
    auto serData = ShaderSerializer::RasterSerialize(
        str.properties,
        kernelArgs,
        vertBin, pixelBin, md5, str.typeMD5, bdlsBufferCount,
        str.printers);
    static_cast<void>(fileIo->write_shader_bytecode(fileName, {reinterpret_cast<std::byte const *>(serData.data()), serData.size_bytes()}));
}
RasterShader *RasterShader::LoadRaster(
    BinaryIO const *fileIo,
    Device *device,
    const MeshFormat &mesh_format,
    luisa::span<Type const *const> types,
    vstd::string_view fileName) {
    vstd::MD5 typeMD5;
    auto ptr = ShaderSerializer::RasterDeSerialize(fileName, CacheType::ByteCode, device, *device->fileIo, {}, typeMD5, mesh_format);
    if (ptr) {
        auto md5 = hlsl::CodegenUtility::GetTypeMD5(types);
        LUISA_ASSERT(md5 == typeMD5, "Shader {} arguments unmatch to requirement!", fileName);
    }
    return ptr;
}
ID3D12PipelineState *RasterShader::GetPSO(
    vstd::span<GFXFormat const> rtvFormats,
    DepthFormat dsvFormat,
    RasterState const &rasterState) {
    std::pair<PSOMap::Index, bool> idx;
    RasterPSOState psoState;
    psoState.rtvFormats.push_back_uninitialized(rtvFormats.size());
    if (!rtvFormats.empty()) {
        memcpy(psoState.rtvFormats.data(), rtvFormats.data(), rtvFormats.size_bytes());
    }
    psoState.dsvFormat = dsvFormat;
    psoState.rasterState = rasterState;
    {
        std::lock_guard lck{psoMtx};
        idx = psoMap.try_emplace(psoState);
    }
    auto &v = idx.first.value();
    return [&]() {
        std::lock_guard lck{v.mtx};
        if (!idx.second) return v.pso.Get();
        vstd::vector<std::byte> md5Bytes;
        auto push = [&]<typename T>(T const &t) {
            auto sz = md5Bytes.size();
            md5Bytes.push_back_uninitialized(sizeof(T));
            memcpy(md5Bytes.data() + sz, &t, sizeof(T));
        };
        auto pushArray = [&]<typename T>(T const *ptr, size_t size) {
            auto sz = md5Bytes.size();
            auto byteSize = size * sizeof(T);
            md5Bytes.push_back_uninitialized(byteSize);
            memcpy(md5Bytes.data() + sz, ptr, byteSize);
        };
        pushArray(psoState.rtvFormats.data(), psoState.rtvFormats.size());
        push(psoState.dsvFormat);
        push(psoState.rasterState);
        push(this->md5);
        auto psoDesc = GetState(
            elements,
            psoState.rasterState,
            rtvFormats,
            dsvFormat);
        psoDesc.pRootSignature = this->rootSig.Get();
        psoDesc.VS = {vertBinData.data(), vertBinData.size()};
        psoDesc.PS = {pixelBinData.data(), pixelBinData.size()};
        auto psoMD5 = vstd::MD5{vstd::span<const uint8_t>{reinterpret_cast<uint8_t const *>(md5Bytes.data()), md5Bytes.size()}};
        auto psoName = psoMD5.to_string(false);
        auto psoStream = device->fileIo->read_shader_cache(psoName);
        auto createPipe = [&] {
            return device->device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(v.pso.GetAddressOf()));
        };
        // use PSO cache
        bool newPso = false;
        if (psoStream != nullptr && psoStream->length() > 0) {
            vstd::vector<std::byte> psoCode;
            psoCode.push_back_uninitialized(psoStream->length());
            psoStream->read({psoCode.data(), psoCode.size()});
            psoDesc.CachedPSO = {
                .pCachedBlob = psoCode.data(),
                .CachedBlobSizeInBytes = psoCode.size()};
            auto psoGenSuccess = createPipe();
            if (psoGenSuccess != S_OK) {
                newPso = true;
                // PSO cache miss(probably driver's version or hardware transformed), discard cache
                LUISA_VERBOSE("{} pipeline cache illegal, discarded.", psoName);
                if (v.pso == nullptr) {
                    psoDesc.CachedPSO.CachedBlobSizeInBytes = 0;
                    psoDesc.CachedPSO.pCachedBlob = nullptr;
                    ThrowIfFailed(createPipe());
                }
            }
        } else {
            newPso = true;
            psoDesc.CachedPSO.pCachedBlob = nullptr;
            ThrowIfFailed(createPipe());
        }
        if (newPso) {
            SavePSO(v.pso.Get(), psoName, device->fileIo, device);
        }
        return v.pso.Get();
    }();
}
// Prepared for indirect
// ID3D12CommandSignature *RasterShader::CmdSig(size_t vertexCount, bool index) {
//     std::lock_guard lck(cmdSigMtx);
//     auto ite = cmdSigs.try_emplace(std::pair<size_t, bool>(vertexCount, index));
//     auto &cmd = ite.first->second;
//     if (!ite.second) return cmd.Get();
//     vstd::vector<D3D12_INDIRECT_ARGUMENT_DESC> indDesc;
//     indDesc.reserve(vertexCount + (index ? 1 : 0) + 2);
//     size_t byteSize = 4 + vertexCount * sizeof(D3D12_VERTEX_BUFFER_VIEW) + (index ? (sizeof(D3D12_DRAW_INDEXED_ARGUMENTS) + sizeof(D3D12_INDEX_BUFFER_VIEW)) : sizeof(D3D12_DRAW_ARGUMENTS));
//     {
//         auto &cst = indDesc.emplace_back();
//         cst.Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
//         cst.Constant.RootParameterIndex = properties.size();
//         cst.Constant.DestOffsetIn32BitValues = 0;
//         cst.Constant.Num32BitValuesToSet = 1;
//     }
//     for (auto &&i : vstd::range(vertexCount)) {
//         auto &vbv = indDesc.emplace_back();
//         vbv.Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
//         vbv.VertexBuffer.Slot = i;
//     }
//     if (index) {
//         auto &idv = indDesc.emplace_back();
//         idv.Type = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;
//         auto &draw = indDesc.emplace_back();
//         draw.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
//     } else {
//         auto &draw = indDesc.emplace_back();
//         draw.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
//     }
//     D3D12_COMMAND_SIGNATURE_DESC desc{
//         .ByteStride = static_cast<uint>(byteSize),
//         .NumArgumentDescs = static_cast<uint>(indDesc.size()),
//         .pArgumentDescs = indDesc.data()};
//     ThrowIfFailed(device->device->CreateCommandSignature(&desc, rootSig.Get(), IID_PPV_ARGS(&cmd)));
//     return cmd.Get();
// }
}// namespace lc::dx
