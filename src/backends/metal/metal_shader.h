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
    id<MTLArgumentEncoder> _encoder{nullptr};
    NSArray<MTLStructMember *> *_arguments{nullptr};

public:
    MetalShader() noexcept = default;
    MetalShader(id<MTLComputePipelineState> pso,
                id<MTLArgumentEncoder> encoder,
                NSArray<MTLStructMember *> *arguments) noexcept
        : _handle{pso},
          _encoder{encoder},
          _arguments{arguments} {}
    ~MetalShader() noexcept {
        _handle = nullptr;
        _encoder = nullptr;
        _arguments = nullptr;
    }

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto encoder() const noexcept { return _encoder; }
    [[nodiscard]] auto arguments() const noexcept { return _arguments; }
};

}// namespace luisa::compute::metal
