#pragma once
#include <Shader/Shader.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/core/binary_io.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/vstl/hash.h>
namespace lc::hlsl {
struct CodegenResult;
}
namespace lc::dx {
class ShaderSerializer;
struct RasterPSOState {
    vstd::vector<GFXFormat> rtvFormats;
    DepthFormat dsvFormat;
    RasterState rasterState;
};
struct RasterPSOStateHash {
    size_t operator()(RasterPSOState const &v) const {
        size_t hash;
        if (v.rtvFormats.empty()) {
            hash = luisa::hash64_default_seed;
        } else {
            hash = luisa::hash64(v.rtvFormats.data(), v.rtvFormats.size_bytes(), luisa::hash64_default_seed);
        }
        hash = luisa::hash64(&v.dsvFormat, sizeof(v.dsvFormat), hash);
        hash = luisa::hash64(&v.rasterState, sizeof(v.rasterState), hash);
        return hash;
    }
};
struct RasterPSOStateEqual {
    int32_t operator()(RasterPSOState const &a, RasterPSOState const &b) const {
        auto rtvSizeComp = vstd::compare<size_t>{}(a.rtvFormats.size(), b.rtvFormats.size());
        if (rtvSizeComp != 0) return rtvSizeComp;
        if (!a.rtvFormats.empty()) {
            auto level = memcmp(a.rtvFormats.data(), b.rtvFormats.data(), a.rtvFormats.size_bytes());
            if (level != 0) return level;
        }
        auto dsvComp = vstd::compare<DepthFormat>{}(a.dsvFormat, b.dsvFormat);
        if (dsvComp != 0)
            return dsvComp;
        return memcmp(&a.rasterState, &b.rasterState, sizeof(RasterState));
    }
};
class RasterShader final : public Shader {
    friend class ShaderSerializer;

private:
    Device *device;
    vstd::MD5 md5;
    vstd::vector<std::byte> vertBinData;
    vstd::vector<std::byte> pixelBinData;
    vstd::vector<D3D12_INPUT_ELEMENT_DESC> elements;
    RasterShader(
        Device *device,
        vstd::MD5 md5,
        MeshFormat const &meshFormat,
        vstd::vector<hlsl::Property> &&prop,
        vstd::vector<SavedArgument> &&args,
        ComPtr<ID3D12RootSignature> &&rootSig,
        vstd::vector<std::pair<vstd::string, Type const*>>&& printers,
        vstd::vector<std::byte> &&vertBinData,
        vstd::vector<std::byte> &&pixelBinData);
    std::mutex psoMtx;
    struct PsoValue {
        ComPtr<ID3D12PipelineState> pso{};
        std::mutex mtx;
    };
    using PSOMap = vstd::HashMap<RasterPSOState, PsoValue, RasterPSOStateHash, RasterPSOStateEqual>;
    PSOMap psoMap;

    // Prepared for indirect

    // struct PairEqual {
    //     using type = std::pair<size_t, bool>;
    //     bool operator()(type const &a, type const &b) const {
    //         return a.first == b.first && a.second == b.second;
    //     }
    // };
    // mutable luisa::unordered_map<
    //     std::pair<size_t, bool>,
    //     ComPtr<ID3D12CommandSignature>,
    //     vstd::hash<std::pair<size_t, bool>>,
    //     PairEqual>
    //     cmdSigs;
    // mutable std::mutex cmdSigMtx;

public:
    // ID3D12CommandSignature *CmdSig(size_t vertexCount, bool index);
    ID3D12PipelineState *GetPSO(
        vstd::span<GFXFormat const> rtvFormats,
        DepthFormat dsvFormat,
        RasterState const &rasterState);
    Tag GetTag() const noexcept override { return Tag::RasterShader; }
    static vstd::MD5 GenMD5(
        vstd::MD5 const &codeMD5,
        MeshFormat const &meshFormat);
    static void GetMeshFormatState(
        vstd::vector<D3D12_INPUT_ELEMENT_DESC> &inputLayout,
        MeshFormat const &meshFormat);
    static D3D12_GRAPHICS_PIPELINE_STATE_DESC GetState(
        vstd::span<D3D12_INPUT_ELEMENT_DESC const> inputLayout,
        RasterState const &state,
        vstd::span<GFXFormat const> rtv,
        DepthFormat dsv);
    RasterShader(
        Device *device,
        vstd::MD5 md5,
        vstd::vector<hlsl::Property> &&prop,
        vstd::vector<SavedArgument> &&args,
        MeshFormat const &meshFormat,
        vstd::vector<std::pair<vstd::string, Type const*>>&& printers,
        vstd::vector<std::byte> &&vertBinData,
        vstd::vector<std::byte> &&pixelBinData);

    ~RasterShader();

    static RasterShader *CompileRaster(
        luisa::BinaryIO const *fileIo,
        Device *device,
        Function vertexKernel,
        Function pixelKernel,
        vstd::function<hlsl::CodegenResult()> const &codegen,
        vstd::MD5 const &md5,
        uint shaderModel,
        MeshFormat const &meshFormat,
        vstd::string_view fileName,
        CacheType cacheType,
        bool enableUnsafeMath);
    static void SaveRaster(
        luisa::BinaryIO const *fileIo,
        Device *device,
        hlsl::CodegenResult const &result,
        vstd::MD5 const &md5,
        vstd::string_view fileName,
        Function vertexKernel,
        Function pixelKernel,
        uint shaderModel,
        bool enableUnsafeMath);
    static RasterShader *LoadRaster(
        luisa::BinaryIO const *fileIo,
        Device *device,
        MeshFormat const &meshFormat,
        luisa::span<Type const *const> types,
        vstd::string_view fileName);
};
}// namespace lc::dx
