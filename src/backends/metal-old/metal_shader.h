//
// Created by Mike Smith on 2021/7/6.
//

#pragma once

#import <Metal/Metal.h>
#import <core/basic_types.h>

namespace luisa::compute::metal {

class alignas(16) MetalShader {

private:
    id<MTLComputePipelineState> _handle{nullptr};

public:
    MetalShader() noexcept = default;
    explicit MetalShader(id<MTLComputePipelineState> pso) noexcept : _handle{pso} {}
    ~MetalShader() noexcept { _handle = nullptr; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
};

}// namespace luisa::compute::metal
