#pragma once
#include <Shader/Shader.h>
#include <runtime/raster/raster_state.h>
#include <runtime/rhi/pixel.h>
#include <core/binary_io.h>
#include <core/stl/unordered_map.h>
#include <vstl/hash.h>
namespace toolhub::directx {
struct CodegenResult;
class ShaderSerializer;
class RasterShader final : public Shader {
    friend class ShaderSerializer;

private:
    Device *device;
    TopologyType type;
    RasterShader(
        Device *device,
        vstd::vector<Property> &&prop,
        vstd::vector<SavedArgument> &&args,
        ComPtr<ID3D12RootSignature> &&rootSig,
        ComPtr<ID3D12PipelineState> &&pso,
        TopologyType type);
    struct PairEqual {
        using type = std::pair<size_t, bool>;
        bool operator()(type const &a, type const &b) const {
            return a.first == b.first && a.second == b.second;
        }
    };
    mutable luisa::unordered_map<
        std::pair<size_t, bool>,
        ComPtr<ID3D12CommandSignature>,
        vstd::hash<std::pair<size_t, bool>>,
        PairEqual>
        cmdSigs;
    mutable std::mutex cmdSigMtx;

public:
    ID3D12CommandSignature *CmdSig(size_t vertexCount, bool index);
    TopologyType TopoType() const { return type; }
    Tag GetTag() const noexcept override { return Tag::RasterShader; }
    static vstd::MD5 GenMD5(
        vstd::MD5 const &codeMD5,
        MeshFormat const &meshFormat,
        RasterState const &state,
        vstd::span<PixelFormat const> rtv,
        DepthFormat dsv);
    static void GetMeshFormatState(
        vstd::vector<D3D12_INPUT_ELEMENT_DESC> &inputLayout,
        MeshFormat const &meshFormat);
    static D3D12_GRAPHICS_PIPELINE_STATE_DESC GetState(
        vstd::vector<D3D12_INPUT_ELEMENT_DESC> &inputLayout,
        MeshFormat const &meshFormat,
        RasterState const &state,
        vstd::span<PixelFormat const> rtv,
        DepthFormat dsv);
    RasterShader(
        Device *device,
        vstd::vector<Property> &&prop,
        vstd::vector<SavedArgument> &&args,
        MeshFormat const &meshFormat,
        RasterState const &state,
        vstd::span<PixelFormat const> rtv,
        DepthFormat dsv,
        vstd::span<std::byte const> vertBinData,
        vstd::span<std::byte const> pixelBinData);

    ~RasterShader();

    static RasterShader *CompileRaster(
        luisa::BinaryIO *fileIo,
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
        bool isInternal);
    static void SaveRaster(
        luisa::BinaryIO *fileIo,
        Device *device,
        CodegenResult const &result,
        vstd::MD5 const &md5,
        vstd::string_view fileName,
        Function vertexKernel,
        Function pixelKernel,
        uint shaderModel);
    static RasterShader *LoadRaster(
        luisa::BinaryIO *fileIo,
        Device *device,
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        luisa::span<Type const *const> types,
        vstd::string_view fileName);
};
}// namespace toolhub::directx