//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <core/stl/lru_cache.h>
#include <runtime/rhi/resource.h>
#include <backends/metal/metal_api.h>

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
    _load_disk_archive(uint64_t hash, luisa::string_view name,
                       const ShaderOption &option, uint3 block_size) const noexcept;

    void _store_disk_archive(uint64_t hash, luisa::string_view name,
                             const ShaderOption &option, uint3 block_size,
                             MTL::ComputePipelineDescriptor *pipeline_desc) const noexcept;

    [[nodiscard]] std::pair<NS::SharedPtr<MTL::ComputePipelineDescriptor>,
                            NS::SharedPtr<MTL::ComputePipelineState>>
    _load_kernel_from_library(MTL::Library *library, luisa::string_view name,
                              const ShaderOption &option, uint3 block_size) const noexcept;

public:
    explicit MetalCompiler(const MetalDevice *device) noexcept;
    [[nodiscard]] NS::SharedPtr<MTL::ComputePipelineState> compile(
        luisa::string_view src, const ShaderOption &option, uint3 block_size) const noexcept;
};

}// namespace luisa::compute::metal
