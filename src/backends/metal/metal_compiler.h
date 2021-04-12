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

namespace luisa::compute::metal {

class MetalDevice;

class MetalCompiler {

public:
    struct alignas(16) KernelItem {
        id<MTLComputePipelineState> handle;
        id<MTLArgumentEncoder> encoder;
        NSArray<MTLStructMember *> *arguments;
        KernelItem(id<MTLComputePipelineState> pso,
                   id<MTLArgumentEncoder> encoder,
                   NSArray<MTLStructMember *> *arguments) noexcept
            : handle{pso},
              encoder{encoder},
              arguments{arguments} {}
    };

private:
    MetalDevice *_device;
    std::unordered_map<uint64_t, KernelItem> _cache;
    std::unordered_map<uint32_t, KernelItem> _kernels;
    spin_mutex _cache_mutex;
    spin_mutex _kernel_mutex;

private:
    [[nodiscard]] KernelItem _compile(uint32_t uid) noexcept;

public:
    explicit MetalCompiler(MetalDevice *device) noexcept : _device{device} {}
    [[nodiscard]] KernelItem kernel(uint32_t uid) noexcept;
};

}
