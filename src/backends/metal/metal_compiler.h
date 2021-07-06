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
#import <backends/metal/metal_shader.h>

namespace luisa::compute::metal {

class MetalDevice;

class MetalCompiler {

private:
    MetalDevice *_device;
    std::unordered_map<uint64_t, MetalShader> _cache;
    spin_mutex _cache_mutex;

public:
    explicit MetalCompiler(MetalDevice *device) noexcept : _device{device} {}
    [[nodiscard]] MetalShader compile(Function kernel) noexcept;
};

}// namespace luisa::compute::metal
