#pragma once

#include <luisa/core/stl/lru_cache.h>
#include <luisa/runtime/rhi/resource.h>
#include "metal_api.h"
#include "metal_shader.h"
#include "metal_shader_metadata.h"

namespace luisa::compute::metal {

class MetalDevice;

class MetalCompiler {

public:
    using Cache = LRUCache<uint64_t /* hash */,
                           MetalShaderHandle>;
    static constexpr auto max_cache_item_count = 64u;

private:
    struct PipelineDescriptorHandle {
        NS::SharedPtr<MTL::ComputePipelineDescriptor> entry;
        NS::SharedPtr<MTL::ComputePipelineDescriptor> indirect_entry;
    };

private:
    const MetalDevice *_device;
    mutable Cache _cache;

private:
    [[nodiscard]] MetalShaderHandle
    _load_disk_archive(luisa::string_view name, bool is_aot,
                       MetalShaderMetadata &metadata) const noexcept;

    void _store_disk_archive(luisa::string_view name, bool is_aot,
                             const PipelineDescriptorHandle &desc,
                             const MetalShaderMetadata &metadata) const noexcept;

    [[nodiscard]] std::pair<PipelineDescriptorHandle, MetalShaderHandle>
    _load_kernels_from_library(MTL::Library *library, uint3 block_size) const noexcept;

public:
    explicit MetalCompiler(const MetalDevice *device) noexcept;

    [[nodiscard]] MetalShaderHandle compile(
        luisa::string_view src, const ShaderOption &option,
        MetalShaderMetadata &metadata,
        MTL::LanguageVersion minimum_version = MTL::LanguageVersion3_0,
        luisa::string *diagnostic = nullptr) const noexcept;

    [[nodiscard]] MetalShaderHandle load(
        luisa::string_view name, MetalShaderMetadata &metadata) const noexcept;

    // Unmodified external device source with a typed direct-buffer ABI. This
    // path shares the device compiler and memory cache, not the AST archive
    // format (which assumes kernel_main + kernel_main_indirect).
    [[nodiscard]] MetalShaderHandle compile_buffer_kernel(
        luisa::string_view src, luisa::string_view entry,
        const ShaderOption &option, MetalShaderMetadata &metadata,
        luisa::string &diagnostic,
        MTL::LanguageVersion minimum_version = MTL::LanguageVersion3_0) const noexcept;
};

}// namespace luisa::compute::metal
