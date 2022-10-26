//
// Created by Mike Smith on 2021/3/24.
//

#pragma once

#import <vector>
#import <future>

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import <core/hash.h>
#import <core/spin_mutex.h>
#import <core/lru_cache.h>
#import <backends/metal/metal_shader.h>

namespace luisa::compute::metal {

class MetalDevice;

class MetalCompiler {

public:
    static constexpr size_t max_cache_item_count = 64u;
    using Cache = LRUCache<uint64_t, MetalShader>;

private:
    MetalDevice *_device;
    luisa::unique_ptr<Cache> _cache;

public:
    explicit MetalCompiler(MetalDevice *device) noexcept
        : _device{device}, _cache{Cache::create(max_cache_item_count)} {}
    MetalCompiler(MetalCompiler &&) noexcept = default;
    MetalCompiler(const MetalCompiler &) noexcept = delete;
    MetalCompiler &operator=(MetalCompiler &&) noexcept = default;
    MetalCompiler &operator=(const MetalCompiler &) noexcept = delete;
    [[nodiscard]] MetalShader compile(Function kernel, std::string_view meta_options) noexcept;
};

}// namespace luisa::compute::metal
