#pragma once
#include "shader.h"
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/raster/raster_state.h>
namespace lc::vk {
class RasterShader : public Shader {
public:
    struct Pipeline {
        VkPipeline pipeline{};
        VkRenderPass render_pass{};
    };
private:
    vstd::vector<uint> _vertex_spv_code;
    vstd::vector<uint> _pixel_spv_code;
    struct PtrHash {
        template<typename T>
        size_t operator()(T const &vec) const {
            return luisa::hash64(vec.data(), vec.size(), luisa::hash64_default_seed);
        }
    };
    struct PtrEqual {
        template<typename A, typename B>
        int operator()(A const &a, B const &b) const {
            if (a.size() > b.size()) { return -1; }
            if (b.size() > a.size()) { return 1; }
            return std::memcmp(a.data(), b.data(), a.size());
        }
    };
    // The key is an opaque byte blob (see `build_raster_pipeline_key`), so the
    // cache hashes and compares the blobs themselves. It used to be a private
    // `BinaryBlob` wrapper around a raw allocation; a `luisa::vector<std::byte>`
    // holds the same bytes in the same order and, being movable, keeps the
    // addresses the vertex-input create-info aliases stable across the insert.
    vstd::HashMap<luisa::vector<std::byte>, Pipeline, PtrHash, PtrEqual> _pipelines;
    VkPipelineCache _pipe_cache{};
    // Forwards to `build_raster_pipeline_key` (raster_pipeline.h), the shared
    // key builder the native raster shader uses as well.
    static luisa::vector<std::byte> _make_pipeline_key(
        luisa::compute::MeshFormat const &mesh_format,
        RasterState const &state,
        luisa::span<Argument::Texture const> rtv_textures,
        Argument::Texture dsv_texture,
        luisa::span<uint const> strides,
        VkPipelineVertexInputStateCreateInfo &vertex_input_create_info);
public:
    RasterShader(
        Device *device,
        vstd::vector<Argument> &&captured,
        vstd::vector<SavedArgument> &&saved_arguments,
        vstd::span<hlsl::Property const> binds,
        vstd::span<std::byte const> cache_code,
        vstd::vector<uint> &&vertex_spv_code,
        vstd::vector<uint> &&pixel_spv_code,
        bool use_tex2d_bindless,
        bool use_tex3d_bindless,
        bool use_buffer_bindless,
        uint validation_count = 0,
        detail::ShaderCodegenDialect codegen_dialect =
            detail::ShaderCodegenDialect::HLSL_SPIRV);
    Pipeline create_pipeline(
        luisa::span<Argument::Texture const> rtv_textures,
        Argument::Texture dsv_textures,
        luisa::compute::MeshFormat const &mesh_format,
        RasterState const &state,
        luisa::span<uint const> strides);
    static VkRenderPass create_render_pass(
        Device *device,
        RasterState const &state,
        luisa::span<Argument::Texture const> rtv_textures,
        Argument::Texture dsv_textures);
    ~RasterShader();
};
}// namespace lc::vk
