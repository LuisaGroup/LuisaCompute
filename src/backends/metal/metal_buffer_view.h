//
// Created by Mike Smith on 2021/6/30.
//

#pragma once

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

namespace luisa::compute::metal {

class alignas(16u) MetalBufferView {

private:
    id<MTLBuffer> _handle;
    uint32_t _offset;
    uint32_t _size;

public:
    MetalBufferView(id<MTLBuffer> handle, size_t offset, size_t size) noexcept
        : _handle{handle},
          _offset{static_cast<uint32_t>(offset)},
          _size{static_cast<uint32_t>(size)} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
};

}
