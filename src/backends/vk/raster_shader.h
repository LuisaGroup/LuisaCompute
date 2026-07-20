#pragma once
#include "shader.h"
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
    struct BinaryBlob {
        auto data() const { return _ptr; }
        auto size() const { return _size; }
        auto size_bytes() const { return _size; }
        BinaryBlob() : _ptr(nullptr), _size(0) {}
        BinaryBlob(size_t size)
            : _ptr(reinterpret_cast<std::byte *>(vengine_malloc(size))),
              _size(size) {
            std::memset(_ptr, 0, size);
        }
        BinaryBlob(BinaryBlob const &) = delete;
        BinaryBlob(BinaryBlob &&rhs) {
            _ptr = rhs._ptr;
            _size = rhs._size;
            rhs._ptr = nullptr;
            rhs._size = 0;
        }
        ~BinaryBlob() {
            if (_ptr) {
                vengine_free(_ptr);
            }
        }
    private:
        std::byte *_ptr;
        size_t _size;
    };

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
    vstd::HashMap<BinaryBlob, Pipeline, PtrHash, PtrEqual> _pipelines;
    VkPipelineCache _pipe_cache{};
    static BinaryBlob _make_pipeline_key(
        luisa::compute::MeshFormat const &mesh_format,
        RasterState const &state,
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
        RasterState const &state);
    static VkRenderPass create_render_pass(
        Device *device,
        RasterState const &state,
        luisa::span<Argument::Texture const> rtv_textures,
        Argument::Texture dsv_textures);
    ~RasterShader();
};
}// namespace lc::vk
