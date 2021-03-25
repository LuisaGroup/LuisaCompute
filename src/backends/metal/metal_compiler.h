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
#import <compile/codegen.h>

namespace luisa::compute::metal {

class MetalCodegen : public compile::Codegen {
public:
    explicit MetalCodegen(Scratch &scratch) noexcept;
    void emit(Function f) override;
};

class MetalCompiler {

public:
    struct PipelineState {
        id<MTLComputePipelineState> handle;
    };
    
    struct alignas(16) KernelItem {
        uint64_t hash;
        PipelineState pso;
        KernelItem(uint64_t hash, id<MTLComputePipelineState> pso) noexcept
            : hash{hash}, pso{pso} {}
    };
    
    struct alignas(16) KernelHandle {
        std::future<PipelineState> pso;
        uint32_t uid;
        KernelHandle(uint32_t uid, std::future<PipelineState> pso) noexcept
            : pso{std::move(pso)}, uid{uid} {}
    };

private:
    id<MTLDevice> _device;
    std::vector<KernelItem> _cache;
    std::vector<KernelHandle> _kernels;
    spin_mutex _cache_mutex;
    spin_mutex _kernel_mutex;
    
private:
    [[nodiscard]] PipelineState _compile(uint32_t uid, std::string_view s) noexcept;

public:
    explicit MetalCompiler(id<MTLDevice> device) noexcept : _device{device} {}
    void prepare(uint32_t uid) noexcept;
    [[nodiscard]] id<MTLComputePipelineState> kernel(uint32_t uid) noexcept;
};

}
