#include <Shader/RasterShader.h>
#include <Resource/DepthBuffer.h>
#include <Shader/ShaderSerializer.h>
#include "../../common/hlsl/hlsl_codegen.h"
#include "../../common/hlsl/shader_compiler.h"
#include <luisa/vstl/md5.h>
#include <luisa/core/logging.h>
namespace lc::dx {
namespace RasterShaderDetail {
static const bool RASTER_PRINT_CODE = ([] {
    // read env LUISA_DUMP_SOURCE
    auto env = std::getenv("LUISA_DUMP_SOURCE");
    if (env == nullptr) {
        return false;
    }
    return std::string_view{env} == "1";
})();
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
    vstd::vector<hlsl::Property> &&prop,
    vstd::vector<SavedArgument> &&args,
    ComPtr<ID3D12RootSignature> &&root_sig,
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    vstd::vector<std::byte> &&vert_bin_data,
    vstd::vector<std::byte> &&pixel_bin_data)
    : Shader(std::move(prop), std::move(args), std::move(root_sig), std::move(printers)), _device(device), _md5{md5},
      _vert_bin_data{std::move(vert_bin_data)}, _pixel_bin_data{std::move(pixel_bin_data)} {
}
void RasterShader::get_mesh_format_state(
    vstd::vector<D3D12_INPUT_ELEMENT_DESC> &inputLayout,
    MeshFormat const &meshFormat) {
    inputLayout.clear();
    inputLayout.reserve(meshFormat.vertex_attribute_count());
    static auto SemanticName = {
        "POSITION",
        "NORMAL",
        "TANGENT",
        "COLOR",
        "TEXCOORD",
        "TEXCOORD",
        "TEXCOORD",
        "TEXCOORD"};
    static auto SemanticIndex = {0u, 0u, 0u, 0u, 0u, 1u, 2u, 3u};
    vstd::fixed_vector<uint, 4> offsets(meshFormat.vertex_stream_count());
    std::memset(offsets.data(), 0, luisa::size_bytes(offsets));
    for (auto i : vstd::range(meshFormat.vertex_stream_count())) {
        auto vec = meshFormat.attributes(i);
        for (auto &&attr : vec) {
            size_t size = pixel_format_size(attr.format, uint3(1, 1, 1));
            auto format = static_cast<DXGI_FORMAT>(TextureBase::ToGFXFormat(attr.format));
            auto &offset = offsets[i];
            offset = CalcAlign(offset, pixel_format_align(attr.format));
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
    vstd::vector<std::pair<vstd::string, Type const *>> &&printers,
    vstd::vector<std::byte> &&vert_bin_data,
    vstd::vector<std::byte> &&pixel_bin_data)
    : Shader(std::move(prop), std::move(args), device->device.Get(), std::move(printers), true),
      _device(device), _md5{md5},
      _vert_bin_data{std::move(vert_bin_data)}, _pixel_bin_data{std::move(pixel_bin_data)} {
}
RasterShader::~RasterShader() {}
D3D12_GRAPHICS_PIPELINE_STATE_DESC RasterShader::get_state(
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
vstd::MD5 RasterShader::gen_md5(
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
        .meshFormatMD5 = vstd_xxhash_gethash(streamHashes.data(), luisa::size_bytes(streamHashes))};
    return vstd::MD5(
        vstd::span<uint8_t const>{
            reinterpret_cast<uint8_t const *>(&h),
            sizeof(Hashes)});
}
RasterShader *RasterShader::compile_raster(
    BinaryIO const *file_io,
    Device *device,
    Function vertexKernel,
    Function pixelKernel,
    vstd::function<hlsl::CodegenResult()> const &codegen,
    vstd::MD5 const &md5,
    uint shaderModel,
    vstd::string_view fileName,
    CacheType cacheType,
    bool enableUnsafeMath,
    bool debug) {
    auto CompileNewCompute = [&](bool writeCache) -> RasterShader * {
        auto str = codegen();
        uint bdlsBufferCount = 0;
        if (str.useBufferBindless) bdlsBufferCount++;
        if (str.useTex2DBindless) bdlsBufferCount++;
        if (str.useTex3DBindless) bdlsBufferCount++;
        if (RasterShaderDetail::RASTER_PRINT_CODE) {
            auto f = fopen("hlsl_output.hlsl", "wb");
            fwrite(str.result.data(), str.result.size(), 1, f);
            fclose(f);
        }
        auto compResult = Device::compiler()->compile_raster(
            str.result.view(),
            true,
            shaderModel, enableUnsafeMath,
            false, debug);
        if (compResult.vertex.is_type_of<vstd::string>()) [[unlikely]] {
            LUISA_ERROR("DXC compile vertex-shader error: {}", compResult.vertex.get<1>());
            return nullptr;
        }
        if (compResult.pixel.is_type_of<vstd::string>()) [[unlikely]] {
            LUISA_ERROR("DXC compile pixel-shader error: {}", compResult.pixel.get<1>());
            return nullptr;
        }
        auto kernelArgs = RasterShaderDetail::GetKernelArgs(vertexKernel, pixelKernel);
        auto GetVector = [&](hlsl::ComUniquePtr<IDxcBlob> &blob) {
            vstd::vector<std::byte> vec;
            luisa::enlarge_by(vec, blob->GetBufferSize());
            std::memcpy(vec.data(), blob->GetBufferPointer(), blob->GetBufferSize());
            return vec;
        };
        auto vertBin = GetVector(compResult.vertex.get<0>());
        auto pixelBin = GetVector(compResult.pixel.get<0>());

        if (writeCache) {
            auto serData = ShaderSerializer::RasterSerialize(
                str.properties,
                kernelArgs, vertBin, pixelBin, md5, str.typeMD5, bdlsBufferCount,
                str.printers);
            write_binary_io(cacheType, file_io, fileName, {reinterpret_cast<std::byte const *>(serData.data()), luisa::size_bytes(serData)});
        }

        auto s = new RasterShader(
            device,
            md5,
            std::move(str.properties),
            std::move(kernelArgs),
            std::move(str.printers),
            std::move(vertBin),
            std::move(pixelBin));
        s->_bindless_count = bdlsBufferCount;
        return s;
    };

    if (!fileName.empty()) {
        // auto psoName = Shader::PSOName(device, fileName);
        auto result = ShaderSerializer::RasterDeSerialize(
            fileName, cacheType, device, *file_io, md5,
            {});
        if (result) {
            return result;
        }
        return CompileNewCompute(true);
    } else {
        return CompileNewCompute(false);
    }
}
void RasterShader::save_raster(
    BinaryIO const *file_io,
    Device *device,
    hlsl::CodegenResult const &result,
    vstd::MD5 const &md5,
    vstd::string_view fileName,
    Function vertexKernel,
    Function pixelKernel,
    uint shaderModel,
    bool enableUnsafeMath,
    bool debug) {
    if (RasterShaderDetail::RASTER_PRINT_CODE) {
        auto f = fopen("hlsl_output.hlsl", "ab");
        fwrite(result.result.data(), result.result.size(), 1, f);
        fclose(f);
    }
    if (ShaderSerializer::CheckMD5(fileName, md5, *file_io)) return;
    auto compiler = Device::compiler();
    if (compiler) {
        auto compResult = compiler->compile_raster(
            result.result.view(),
            true,
            shaderModel,
            enableUnsafeMath,
            false, debug);

        if (compResult.vertex.is_type_of<vstd::string>()) [[unlikely]] {
            LUISA_WARNING("DXC compile vertex-shader error: {}", compResult.vertex.get<1>());
            return;
        }
        if (compResult.pixel.is_type_of<vstd::string>()) [[unlikely]] {
            LUISA_WARNING("DXC compile pixel-shader error: {}", compResult.pixel.get<1>());
            return;
        }
        auto kernelArgs = RasterShaderDetail::GetKernelArgs(vertexKernel, pixelKernel);
        auto GetSpan = [&](hlsl::ComUniquePtr<IDxcBlob> &blob) {
            return vstd::span{
                reinterpret_cast<std::byte const *>(blob->GetBufferPointer()),
                blob->GetBufferSize()};
        };
        auto vertBin = GetSpan(compResult.vertex.get<0>());
        auto pixelBin = GetSpan(compResult.pixel.get<0>());
        uint bdlsBufferCount = 0;
        if (result.useBufferBindless) bdlsBufferCount++;
        if (result.useTex2DBindless) bdlsBufferCount++;
        if (result.useTex3DBindless) bdlsBufferCount++;
        auto serData = ShaderSerializer::RasterSerialize(
            result.properties,
            kernelArgs,
            vertBin, pixelBin, md5, result.typeMD5, bdlsBufferCount,
            result.printers);
        static_cast<void>(file_io->write_shader_bytecode(fileName, {reinterpret_cast<std::byte const *>(serData.data()), luisa::size_bytes(serData)}));
    } else {
        // write HLSL code if compiler not initialized
        static_cast<void>(file_io->write_shader_bytecode(fileName, {reinterpret_cast<std::byte const *>(result.result.data()), result.result.size()}));
    }
}
RasterShader *RasterShader::load_raster(
    BinaryIO const *file_io,
    Device *device,
    luisa::span<Type const *const> types,
    vstd::string_view fileName) {
    auto ptr = ShaderSerializer::RasterDeSerialize(fileName, CacheType::ByteCode, device, *device->file_io, {}, hlsl::CodegenUtility::GetTypeMD5(types));
    return ptr;
}
ID3D12PipelineState *RasterShader::get_pso(
    vstd::span<GFXFormat const> rtvFormats,
    MeshFormat const &meshFormat,
    DepthFormat dsvFormat,
    RasterState const &rasterState) {
    std::pair<PSOMap::Index, bool> idx;
    RasterPSOState psoState;
    luisa::enlarge_by(psoState.rtv_formats, rtvFormats.size());
    if (!rtvFormats.empty()) {
        std::memcpy(psoState.rtv_formats.data(), rtvFormats.data(), rtvFormats.size_bytes());
    }
    psoState.dsv_format = dsvFormat;
    psoState.raster_state = rasterState;
    {
        std::lock_guard lck{_pso_mtx};
        idx = _pso_map.try_emplace(psoState);
    }
    auto &v = idx.first.value();
    std::lock_guard lck{v.mtx};
    if (!idx.second) return v.pso.Get();
    vstd::vector<D3D12_INPUT_ELEMENT_DESC> elements;
    get_mesh_format_state(elements, meshFormat);
    vstd::vector<std::byte> md5Bytes;
    auto push = [&]<typename T>(T const &t) {
        auto sz = md5Bytes.size();
        luisa::enlarge_by(md5Bytes, sizeof(T));
        std::memcpy(md5Bytes.data() + sz, &t, sizeof(T));
    };
    auto pushArray = [&]<typename T>(T const *ptr, size_t size) {
        auto sz = md5Bytes.size();
        auto byteSize = size * sizeof(T);
        luisa::enlarge_by(md5Bytes, byteSize);
        std::memcpy(md5Bytes.data() + sz, ptr, byteSize);
    };
    pushArray(psoState.rtv_formats.data(), psoState.rtv_formats.size());
    push(psoState.dsv_format);
    push(psoState.raster_state);
    push(this->_md5);
    auto psoDesc = get_state(
        elements,
        psoState.raster_state,
        rtvFormats,
        dsvFormat);
    psoDesc.pRootSignature = this->_root_sig.Get();
    psoDesc.VS = {_vert_bin_data.data(), _vert_bin_data.size()};
    psoDesc.PS = {_pixel_bin_data.data(), _pixel_bin_data.size()};

    auto psoMD5 = vstd::MD5{vstd::span<const uint8_t>{reinterpret_cast<uint8_t const *>(md5Bytes.data()), md5Bytes.size()}};
    auto psoName = psoMD5.to_string(false);
    auto psoStream = _device->file_io->read_shader_cache(psoName);
    auto createPipe = [&] {
        return _device->device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(v.pso.GetAddressOf()));
    };
    // use PSO cache
    bool newPso = false;
    if (psoStream != nullptr && psoStream->length() > 0) {
        vstd::vector<std::byte> psoCode;
        luisa::enlarge_by(psoCode, psoStream->length());
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
        _save_pso(v.pso.Get(), psoName, _device->file_io, _device);
    }
    return v.pso.Get();
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
