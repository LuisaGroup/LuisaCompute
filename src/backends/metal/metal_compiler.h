//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <core/stl/lru_cache.h>
#include <runtime/rhi/resource.h>
#include <backends/metal/metal_api.h>
#include <backends/metal/metal_shader.h>
#include <backends/metal/metal_shader_metadata.h>

namespace luisa::compute::metal {

class MetalDevice;

class MetalCompiler {

public:
    using Cache = LRUCache<uint64_t /* hash */,
                           NS::SharedPtr<MTL::ComputePipelineState>>;
    static constexpr auto max_cache_item_count = 64u;

private:
    const MetalDevice *_device;
    mutable Cache _cache;

private:
    [[nodiscard]] NS::SharedPtr<MTL::ComputePipelineState>
    _load_disk_archive(luisa::string_view name, bool is_aot,
                       MetalShaderMetadata &metadata) const noexcept;

    void _store_disk_archive(luisa::string_view name, bool is_aot,
                             MTL::ComputePipelineDescriptor *pipeline_desc,
                             const MetalShaderMetadata &metadata) const noexcept;

    [[nodiscard]] std::pair<NS::SharedPtr<MTL::ComputePipelineDescriptor>,
                            NS::SharedPtr<MTL::ComputePipelineState>>
    _load_kernel_from_library(MTL::Library *library, uint3 block_size) const noexcept;

public:
    explicit MetalCompiler(const MetalDevice *device) noexcept;

    [[nodiscard]] NS::SharedPtr<MTL::ComputePipelineState> compile(
        luisa::string_view src, const ShaderOption &option,
        MetalShaderMetadata &metadata) const noexcept;

    [[nodiscard]] NS::SharedPtr<MTL::ComputePipelineState> load(
        luisa::string_view name, MetalShaderMetadata &metadata) const noexcept;
};

}// namespace luisa::compute::metal
