#pragma once

#include "native_shader.h"
#include "raster_pipeline.h"
#include "resource.h"

#include <cstring>

#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/vstl/hash_map.h>
#include <volk.h>

namespace lc::vk {

// The reflection table and the resource kinds are `luisa::compute` types; name
// them the way every other header of this backend does instead of relying on a
// sibling header's using-directives.
using namespace luisa;
using namespace luisa::compute;

class Device;
struct CommandBufferState;

// A Vulkan graphics pipeline built directly from two native SPIR-V modules
// (native raster shader injection).
//
// This is the raster counterpart of `NativeShader`: it keeps the *pipeline
// creation pattern* of the DSL `RasterShader` (a pipeline cache, an internal
// render pass per attachment configuration, one pipeline per
// (vertex input, raster state, attachment format) key) but builds the
// descriptor set layouts, the pipeline layout and the descriptor pool from the
// two modules' own reflection, because a native module has none of the DSL's
// persisted descriptor ABI (dense local bindings in set 0, the sampler heap, the
// argument buffer, the constant UBO).
//
// Contract:
//  * `bindings` is the *merged* table of both stages, in canonical
//    `(set, binding, kind)` order - it is
//    `NativeRasterShaderCompileResult::bindings`, exactly the table the launcher
//    binds by index.
//  * `push_constant_size` is the application's uniform bytes;
//    `object_id_bytes` (0 or 4) is reserved in front of them for
//    `RasterMesh::object_id`. The pipeline layout's push-constant range is
//    therefore `object_id_bytes + push_constant_size`, while the launcher's
//    uniform blob starts at offset 0 and is pushed at offset `object_id_bytes`.
//  * Both stages share one pipeline layout, so a (set, binding) that both
//    stages declare is bound once, for
//    `VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT`.
//
// Textures, samplers and acceleration structures are rejected by
// `load_raster()` (the same restriction as the compute route), and descriptors
// are allocated per draw from a pool owned by this shader and released when the
// recording command buffer is recycled.
//
// Lifetime, and why it matters here more than for the compute route: a
// `NativeRasterShader` is typically a short-lived RAII object, but its pipeline,
// render passes, pipeline layout, descriptor set layouts and modules must outlive
// every command buffer that has recorded a draw of it. A draw is submitted when
// it is enqueued, so the rule is: synchronize (or otherwise retire) the draws
// before destroying the shader - the same rule the compute `NativeShader`
// documents. A shader destroyed too early is undefined behaviour, and the
// deferred descriptor-set release that keeps the shader's pool alive does not
// extend to the set layouts (`NativeShader::~NativeShader` has the same
// ordering).
//
// Errors during encoding are fatal (`LUISA_ERROR`, which aborts), like every
// other encoding error in this backend: `encode_draw_state`'s
// `VK_NULL_HANDLE`/early returns are defensive fallbacks for the case where that
// policy changes, not a recoverable error path.
class NativeRasterShader final : public Resource {

public:
    // The pipeline key is an opaque byte blob (see `build_raster_pipeline_key`),
    // so the cache hashes and compares the blobs themselves.
    struct BlobHash {
        size_t operator()(luisa::vector<std::byte> const &blob) const noexcept {
            return luisa::hash64(blob.data(), blob.size(), luisa::hash64_default_seed);
        }
    };
    struct BlobEqual {
        int operator()(luisa::vector<std::byte> const &a,
                       luisa::vector<std::byte> const &b) const noexcept {
            if (a.size() != b.size()) { return a.size() < b.size() ? -1 : 1; }
            return a.empty() ? 0 : std::memcmp(a.data(), b.data(), a.size());
        }
    };

private:
    uint32_t _push_constant_size{0u};// application uniform bytes
    uint32_t _object_id_bytes{0u};   // 0 or 4, reserved in front of them
    VkShaderModule _vertex_module{VK_NULL_HANDLE};
    VkShaderModule _pixel_module{VK_NULL_HANDLE};
    luisa::string _vertex_entry_point;
    luisa::string _pixel_entry_point;
    luisa::vector<NativeShaderResourceBinding> _bindings;
    vstd::vector<NativeShader::LayoutBinding> _layout_bindings;
    vstd::vector<VkDescriptorSetLayout> _set_layouts;
    luisa::shared_ptr<NativeShaderDescriptorPool> _descriptor_pool;
    VkPipelineLayout _pipeline_layout{VK_NULL_HANDLE};
    uint32_t _descriptor_set_count{0u};
    VkPipelineCache _pipeline_cache{VK_NULL_HANDLE};
    vstd::HashMap<luisa::vector<std::byte>, RasterPipelineHandle,
                  BlobHash, BlobEqual>
        _pipelines;

public:
    // Number of descriptor sets a single frame may allocate before the pool is
    // exhausted; the pool is refilled as frames retire. Mirrors
    // `NativeShader::max_in_flight_descriptor_sets`.
    static constexpr auto max_in_flight_descriptor_sets = 256u;

    NativeRasterShader(
        Device *device,
        luisa::span<const NativeShaderResourceBinding> bindings,
        luisa::span<const std::byte> vertex_spirv,
        luisa::string_view vertex_entry_point,
        luisa::span<const std::byte> pixel_spirv,
        luisa::string_view pixel_entry_point,
        uint32_t push_constant_size,
        uint32_t object_id_bytes) noexcept;
    ~NativeRasterShader() noexcept override;
    NativeRasterShader(NativeRasterShader const &) = delete;
    NativeRasterShader(NativeRasterShader &&) = delete;

    [[nodiscard]] uint32_t push_constant_size() const noexcept { return _push_constant_size; }
    [[nodiscard]] uint32_t object_id_bytes() const noexcept { return _object_id_bytes; }
    [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept { return _pipeline_layout; }
    [[nodiscard]] luisa::span<const NativeShaderResourceBinding> bindings() const noexcept {
        return _bindings;
    }

    // Returns the pipeline (and the render pass it was built for) of this draw,
    // creating and caching it on first use. `strides` must carry one entry per
    // vertex stream of `mesh_format`.
    [[nodiscard]] RasterPipelineHandle create_pipeline(
        luisa::span<luisa::compute::Argument::Texture const> rtv_textures,
        luisa::compute::Argument::Texture dsv_texture,
        luisa::compute::MeshFormat const &mesh_format,
        luisa::compute::RasterState const &state,
        luisa::span<uint const> strides) noexcept;

    // Pushes one mesh's object id into the reserved slot. Only valid when
    // `object_id_bytes() == 4`.
    void push_object_id(VkCommandBuffer cmdbuffer, uint32_t object_id) noexcept;

    // Binds the pipeline, allocates and writes this draw's descriptor sets, and
    // pushes the application's uniform block. The caller then records the
    // framebuffer, viewport/scissor, vertex buffers and draws; per-mesh object
    // ids go through `push_object_id()`.
    //
    // Returns the render pass the pipeline was built for - the draw's
    // framebuffer must be created from it - or `VK_NULL_HANDLE` when the
    // pipeline could not be created (already reported).
    [[nodiscard]] VkRenderPass encode_draw_state(
        CommandBufferState *state, VkCommandBuffer cmdbuffer,
        luisa::compute::DrawRasterSceneCommand const *cmd,
        luisa::span<uint const> strides) noexcept;
};

}// namespace lc::vk
