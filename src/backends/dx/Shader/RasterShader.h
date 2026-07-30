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
    vstd::vector<GFXFormat> rtv_formats;
    DepthFormat dsv_format;
    RasterState raster_state;
};
struct RasterPSOStateHash {
    size_t operator()(RasterPSOState const &v) const {
        size_t hash;
        if (v.rtv_formats.empty()) {
            hash = luisa::hash64_default_seed;
        } else {
            hash = luisa::hash64(v.rtv_formats.data(), luisa::size_bytes(v.rtv_formats), luisa::hash64_default_seed);
        }
        hash = luisa::hash64(&v.dsv_format, sizeof(v.dsv_format), hash);
        hash = luisa::hash64(&v.raster_state, sizeof(v.raster_state), hash);
        return hash;
    }
};
struct RasterPSOStateEqual {
    int32_t operator()(RasterPSOState const &a, RasterPSOState const &b) const {
        auto rtvSizeComp = vstd::compare<size_t>{}(a.rtv_formats.size(), b.rtv_formats.size());
        if (rtvSizeComp != 0) return rtvSizeComp;
        if (!a.rtv_formats.empty()) {
            auto level = std::memcmp(a.rtv_formats.data(), b.rtv_formats.data(), luisa::size_bytes(a.rtv_formats));
            if (level != 0) return level;
        }
        auto dsvComp = vstd::compare<DepthFormat>{}(a.dsv_format, b.dsv_format);
        if (dsvComp != 0)
            return dsvComp;
        return std::memcmp(&a.raster_state, &b.raster_state, sizeof(RasterState));
    }
};
class RasterShader final : public Shader {
    friend class ShaderSerializer;

private:
    Device *_device;
    vstd::MD5 _md5;
    vstd::vector<std::byte> _vert_bin_data;
    vstd::vector<std::byte> _pixel_bin_data;
    RasterShader(
        Device *device,
        vstd::MD5 md5,
        vstd::vector<hlsl::Property> &&prop,
        vstd::vector<SavedArgument> &&args,
        ComPtr<ID3D12RootSignature> &&root_sig,
        vstd::vector<std::pair<vstd::string, Type const*>>&& printers,
        vstd::vector<std::byte> &&vert_bin_data,
        vstd::vector<std::byte> &&pixel_bin_data,
        uint validation_count);
    std::mutex _pso_mtx;
    struct PsoValue {
        ComPtr<ID3D12PipelineState> pso{};
        vstd::spin_mutex mtx;
    };
    using PSOMap = vstd::HashMap<RasterPSOState, PsoValue, RasterPSOStateHash, RasterPSOStateEqual>;
    PSOMap _pso_map;

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
    ID3D12PipelineState *get_pso(
        vstd::span<GFXFormat const> rtvFormats,
        MeshFormat const &meshFormat,
        DepthFormat dsvFormat,
        RasterState const &rasterState);
    Tag get_tag() const noexcept override { return Tag::RasterShader; }
    static vstd::MD5 gen_md5(
        vstd::MD5 const &codeMD5,
        MeshFormat const &meshFormat);
    static void get_mesh_format_state(
        vstd::vector<D3D12_INPUT_ELEMENT_DESC> &inputLayout,
        MeshFormat const &meshFormat);
    static D3D12_GRAPHICS_PIPELINE_STATE_DESC get_state(
        vstd::span<D3D12_INPUT_ELEMENT_DESC const> inputLayout,
        RasterState const &state,
        vstd::span<GFXFormat const> rtv,
        DepthFormat dsv);
    RasterShader(
        Device *device,
        vstd::MD5 md5,
        vstd::vector<hlsl::Property> &&prop,
        vstd::vector<SavedArgument> &&args,
        vstd::vector<std::pair<vstd::string, Type const*>>&& printers,
        vstd::vector<std::byte> &&vert_bin_data,
        vstd::vector<std::byte> &&pixel_bin_data,
        uint validation_count);

    ~RasterShader();

    static RasterShader *compile_raster(
        luisa::BinaryIO const *file_io,
        Device *device,
        Function vertexKernel,
        Function pixelKernel,
        vstd::function<hlsl::CodegenResult()> const &codegen,
        vstd::MD5 const &md5,
        uint shaderModel,
        vstd::string_view fileName,
        CacheType cacheType,
        bool enableUnsafeMath,
        bool debug);
    static void save_raster(
        luisa::BinaryIO const *file_io,
        Device *device,
        hlsl::CodegenResult const &result,
        vstd::MD5 const &md5,
        vstd::string_view fileName,
        Function vertexKernel,
        Function pixelKernel,
        uint shaderModel,
        bool enableUnsafeMath,
        bool debug);
    static RasterShader *load_raster(
        luisa::BinaryIO const *file_io,
        Device *device,
        luisa::span<Type const *const> types,
        vstd::string_view fileName);
};
}// namespace lc::dx
