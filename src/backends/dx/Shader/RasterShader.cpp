#include <Shader/RasterShader.h>
#include <Resource/DepthBuffer.h>
#include <Shader/ShaderSerializer.h>
#include <vstl/binary_reader.h>
#include <HLSL/dx_codegen.h>
#include <Shader/ShaderCompiler.h>
#include <vstl/md5.h>
namespace toolhub::directx {
namespace RasterShaderDetail {
static constexpr bool PRINT_CODE = false;
static vstd::vector<SavedArgument> GetKernelArgs(Function vertexKernel, Function pixelKernel) {
    if (vertexKernel.builder() == nullptr || pixelKernel.builder() == nullptr) {
        return {};
    } else {
        auto vertArgs =
            vstd::CacheEndRange(vertexKernel.arguments()) |
            vstd::TransformRange(
                [&](Variable const &var) {
                    return std::pair<Variable, Usage>{var, vertexKernel.variable_usage(var.uid())};
                });
        auto pixelSpan = pixelKernel.arguments();
        auto pixelArgs =
            vstd::ite_range(pixelSpan.begin() + 1, pixelSpan.end()) |
            vstd::TransformRange(
                [&](Variable const &var) {
                    return std::pair<Variable, Usage>{var, pixelKernel.variable_usage(var.uid())};
                });
        auto args = vstd::RangeImpl(vstd::PairIterator(vertArgs, pixelArgs));
        return ShaderSerializer::SerializeKernel(args);
    }
}
}// namespace RasterShaderDetail
RasterShader::RasterShader(
    Device *device,
    vstd::vector<Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ComPtr<ID3D12RootSignature> &&rootSig,
    ComPtr<ID3D12PipelineState> &&pso,
    TopologyType type)
    : Shader(std::move(prop), std::move(args), std::move(rootSig)), device(device), type(type) {
    this->pso = std::move(pso);
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
        // TODO
    }
}
RasterShader::RasterShader(
    Device *device,
    vstd::vector<Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    MeshFormat const &meshFormat,
    RasterState const &state,
    vstd::span<PixelFormat const> rtv,
    DepthFormat dsv,
    vstd::span<std::byte const> vertBinData,
    vstd::span<std::byte const> pixelBinData)
    : Shader(std::move(prop), std::move(args), device->device.Get(), true),
      device(device),
      type(state.topology) {
    vstd::vector<D3D12_INPUT_ELEMENT_DESC> layouts;
    auto psoDesc = GetState(
        layouts,
        meshFormat,
        state,
        rtv,
        dsv);

    psoDesc.pRootSignature = this->rootSig.Get();
    psoDesc.VS = {.pShaderBytecode = vertBinData.data(), .BytecodeLength = vertBinData.size()};
    psoDesc.PS = {.pShaderBytecode = pixelBinData.data(), .BytecodeLength = pixelBinData.size()};
    ThrowIfFailed(device->device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(pso.GetAddressOf())));
}
RasterShader::~RasterShader() {}
D3D12_GRAPHICS_PIPELINE_STATE_DESC RasterShader::GetState(
    vstd::vector<D3D12_INPUT_ELEMENT_DESC> &inputLayout,
    MeshFormat const &meshFormat,
    RasterState const &state,
    vstd::span<PixelFormat const> rtv,
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

    if (state.blend_state.enableBlend) {
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
    if (state.depth_state.enableDepth) {
        auto &v = state.depth_state;
        depth.DepthEnable = true;
        depth.DepthFunc = ComparisonState(v.comparison);
        depth.DepthWriteMask = v.write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    }
    if (state.stencil_state.enableStencil) {
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
        result.RTVFormats[i] = static_cast<DXGI_FORMAT>(TextureBase::ToGFXFormat(rtv[i]));
    }
    result.DSVFormat = static_cast<DXGI_FORMAT>(DepthBuffer::GetDepthFormat(dsv));
    GetMeshFormatState(inputLayout, meshFormat);
    result.InputLayout = {.pInputElementDescs = inputLayout.data(), .NumElements = static_cast<uint>(inputLayout.size())};
    return result;
}
vstd::MD5 RasterShader::GenMD5(
    vstd::MD5 const &codeMD5,
    MeshFormat const &meshFormat,
    RasterState const &state,
    vstd::span<PixelFormat const> rtv,
    DepthFormat dsv) {
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
        uint64 rasterStateMD5;
        uint64 rtv;
        uint64 dsv;
    };
    Hashes h{
        .codeMd5 = codeMD5,
        .meshFormatMD5 = vstd_xxhash_gethash(streamHashes.data(), streamHashes.size_bytes()),
        .rasterStateMD5 = vstd_xxhash_gethash(&state, sizeof(RasterState)),
        .rtv = vstd_xxhash_gethash(rtv.data(), rtv.size_bytes()),
        .dsv = vstd_xxhash_gethash(&dsv, sizeof(DepthFormat))};
    return vstd::MD5(
        vstd::span<uint8_t const>{
            reinterpret_cast<uint8_t const *>(&h),
            sizeof(Hashes)});
}
RasterShader *RasterShader::CompileRaster(
    BinaryIO *fileIo,
    Device *device,
    Function vertexKernel,
    Function pixelKernel,
    vstd::function<CodegenResult()> const &codegen,
    vstd::MD5 const &md5,
    uint shaderModel,
    MeshFormat const &meshFormat,
    RasterState const &state,
    vstd::span<PixelFormat const> rtv,
    DepthFormat dsv,
    vstd::string_view fileName,
    bool isInternal) {
    vstd::optional<vstd::MD5> psoMd5 = GenMD5(md5, meshFormat, state, rtv, dsv);
    auto CompileNewCompute = [&](bool writeCache, vstd::string_view psoName) -> RasterShader * {
        auto str = codegen();
        if constexpr (RasterShaderDetail::PRINT_CODE) {
            auto f = fopen("hlsl_output.hlsl", "ab");
            fwrite(str.result.data(), str.result.size(), 1, f);
            fclose(f);
        }
        auto compResult = Device::Compiler()->CompileRaster(
            str.result.view(),
            true,
            shaderModel);
        if (compResult.vertex.IsTypeOf<vstd::string>()) {
            std::cout << compResult.vertex.get<1>() << '\n';
            VSTL_ABORT();
            return nullptr;
        }
        if (compResult.pixel.IsTypeOf<vstd::string>()) {
            std::cout << compResult.pixel.get<1>() << '\n';
            VSTL_ABORT();
            return nullptr;
        }
        auto kernelArgs = RasterShaderDetail::GetKernelArgs(vertexKernel, pixelKernel);
        auto GetSpan = [&](DXByteBlob const &blob) {
            return vstd::span<std::byte const>{blob.GetBufferPtr(), blob.GetBufferSize()};
        };
        auto vertBin = GetSpan(*compResult.vertex.get<0>());
        auto pixelBin = GetSpan(*compResult.pixel.get<0>());
        if (writeCache) {
            auto serData = ShaderSerializer::RasterSerialize(str.properties, kernelArgs, vertBin, pixelBin, md5, str.typeMD5, str.bdlsBufferCount);
            if (isInternal) {
                fileIo->write_internal_shader(fileName, {reinterpret_cast<std::byte const *>(serData.data()), serData.size_bytes()});
            } else {
                fileIo->write_shader_bytecode(fileName, {reinterpret_cast<std::byte const *>(serData.data()), serData.size_bytes()});
            }
        }

        auto s = new RasterShader(
            device,
            std::move(str.properties),
            std::move(kernelArgs),
            meshFormat,
            state,
            rtv,
            dsv,
            vertBin,
            pixelBin);
        s->bindlessCount = str.bdlsBufferCount;
        if (writeCache) {
            s->SavePSO(psoName, fileIo, device);
        }
        return s;
    };

    if (!fileName.empty()) {
        auto psoName = Shader::PSOName(device, fileName);
        bool oldDeleted = false;
        vstd::MD5 typeMD5;
        auto result = ShaderSerializer::RasterDeSerialize(
            fileName, psoName, isInternal, device, *fileIo, md5,
            psoMd5, typeMD5, meshFormat, state, rtv, dsv, oldDeleted);
        if (result) {
            if (oldDeleted) {
                result->SavePSO(psoName, fileIo, device);
            }
        }
        return CompileNewCompute(true, psoName);
    } else {
        return CompileNewCompute(false, {});
    }
}
void RasterShader::SaveRaster(
    BinaryIO *fileIo,
    Device *device,
    CodegenResult const &str,
    vstd::MD5 const &md5,
    vstd::string_view fileName,
    Function vertexKernel,
    Function pixelKernel,
    uint shaderModel) {
    if constexpr (RasterShaderDetail::PRINT_CODE) {
        auto f = fopen("hlsl_output.hlsl", "ab");
        fwrite(str.result.data(), str.result.size(), 1, f);
        fclose(f);
    }
    if (ShaderSerializer::CheckMD5(fileName, md5, *fileIo)) return;
    auto compResult = Device::Compiler()->CompileRaster(
        str.result.view(),
        true,
        shaderModel);
    if (compResult.vertex.IsTypeOf<vstd::string>()) {
        std::cout << compResult.vertex.get<1>() << '\n';
        VSTL_ABORT();
        return;
    }
    if (compResult.pixel.IsTypeOf<vstd::string>()) {
        std::cout << compResult.pixel.get<1>() << '\n';
        VSTL_ABORT();
        return;
    }
    auto kernelArgs = RasterShaderDetail::GetKernelArgs(vertexKernel, pixelKernel);
    auto GetSpan = [&](DXByteBlob const &blob) {
        return vstd::span<std::byte const>{blob.GetBufferPtr(), blob.GetBufferSize()};
    };
    auto vertBin = GetSpan(*compResult.vertex.get<0>());
    auto pixelBin = GetSpan(*compResult.pixel.get<0>());
    auto serData = ShaderSerializer::RasterSerialize(str.properties, kernelArgs, vertBin, pixelBin, md5, str.typeMD5, str.bdlsBufferCount);
    fileIo->write_shader_bytecode(fileName, {reinterpret_cast<std::byte const *>(serData.data()), serData.size_bytes()});
}
RasterShader *RasterShader::LoadRaster(
    BinaryIO *fileIo,
    Device *device,
    const MeshFormat &mesh_format,
    const RasterState &raster_state,
    luisa::span<const PixelFormat> rtv_format,
    DepthFormat dsv_format,
    luisa::span<Type const *const> types,
    vstd::string_view fileName) {
    auto psoName = Shader::PSOName(device, fileName);

    vstd::optional<vstd::MD5> md5;
    bool cacheDeleted = false;
    vstd::MD5 typeMD5;
    auto ptr = ShaderSerializer::RasterDeSerialize(fileName, psoName, false, device, *device->fileIo, {}, md5, typeMD5, mesh_format, raster_state, rtv_format, dsv_format, cacheDeleted);
    if (ptr) {
        auto md5 = CodegenUtility::GetTypeMD5(types);
        if (md5 != typeMD5) {
            LUISA_ERROR("Shader {} arguments unmatch to requirement!", fileName);
        }

        if (cacheDeleted) {
            ptr->SavePSO(
                psoName,
                fileIo,
                device);
        }
    }
    return ptr;
}
ID3D12CommandSignature *RasterShader::CmdSig(size_t vertexCount, bool index) {
    std::lock_guard lck(cmdSigMtx);
    auto ite = cmdSigs.try_emplace(std::pair<size_t, bool>(vertexCount, index));
    auto &cmd = ite.first->second;
    if (!ite.second) return cmd.Get();
    vstd::vector<D3D12_INDIRECT_ARGUMENT_DESC> indDesc;
    indDesc.reserve(vertexCount + (index ? 1 : 0) + 2);
    size_t byteSize = 4 + vertexCount * sizeof(D3D12_VERTEX_BUFFER_VIEW) + (index ? (sizeof(D3D12_DRAW_INDEXED_ARGUMENTS) + sizeof(D3D12_INDEX_BUFFER_VIEW)) : sizeof(D3D12_DRAW_ARGUMENTS));
    {
        auto &cst = indDesc.emplace_back();
        cst.Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        cst.Constant.RootParameterIndex = properties.size();
        cst.Constant.DestOffsetIn32BitValues = 0;
        cst.Constant.Num32BitValuesToSet = 1;
    }
    for (auto &&i : vstd::range(vertexCount)) {
        auto &vbv = indDesc.emplace_back();
        vbv.Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
        vbv.VertexBuffer.Slot = i;
    }
    if (index) {
        auto &idv = indDesc.emplace_back();
        idv.Type = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;
        auto &draw = indDesc.emplace_back();
        draw.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
    } else {
        auto &draw = indDesc.emplace_back();
        draw.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    }
    D3D12_COMMAND_SIGNATURE_DESC desc{
        .ByteStride = static_cast<uint>(byteSize),
        .NumArgumentDescs = static_cast<uint>(indDesc.size()),
        .pArgumentDescs = indDesc.data()};
    ThrowIfFailed(device->device->CreateCommandSignature(&desc, rootSig.Get(), IID_PPV_ARGS(&cmd)));
    return cmd.Get();
}
}// namespace toolhub::directx