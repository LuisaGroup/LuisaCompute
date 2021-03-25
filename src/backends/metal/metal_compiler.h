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
    
    struct alignas(16) KernelCacheItem {
        uint64_t hash;
        id<MTLComputePipelineState> pso;
        KernelCacheItem(uint64_t hash, id<MTLComputePipelineState> pso) noexcept
            : hash{hash}, pso{pso} {}
    };
    
    struct PipelineState {
        id<MTLComputePipelineState> handle;
    };
    
    struct alignas(16) KernelHandle {
        std::future<PipelineState> pso;
        uint32_t uid;
        KernelHandle(uint32_t uid, std::future<PipelineState> pso) noexcept
            : pso{std::move(pso)}, uid{uid} {}
    };

private:
    MetalDevice *_device;
    std::vector<KernelCacheItem> _cache;
    std::vector<KernelHandle> _kernels;
    spin_mutex _cache_mutex;
    spin_mutex _kernel_mutex;
    
private:
    [[nodiscard]] id<MTLComputePipelineState> _compile(uint32_t uid) noexcept;

public:
    explicit MetalCompiler(MetalDevice *device) noexcept : _device{device} {}
    void prepare(uint32_t uid) noexcept;
    [[nodiscard]] id<MTLComputePipelineState> kernel(uint32_t uid) noexcept;
};

}
