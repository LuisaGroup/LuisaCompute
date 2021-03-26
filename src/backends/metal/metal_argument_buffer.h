//
// Created by Mike Smith on 2021/3/26.
//

#pragma once

#import <vector>

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import <core/spin_mutex.h>

namespace luisa::compute::metal {

class MetalArgumentBuffer {

public:
    static constexpr auto size = 4096u;

private:
    id<MTLBuffer> _handle;
    size_t _offset;

public:
    MetalArgumentBuffer(id<MTLBuffer> handle, size_t offset) noexcept
        : _handle{handle}, _offset{offset} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
};

class MetalArgumentBufferPool {

public:
    static constexpr auto trunk_size = 16u;

private:
    __weak id<MTLDevice> _device;
    std::vector<MetalArgumentBuffer> _available_buffers;
    spin_mutex _mutex;

private:
    void _create_new_trunk_if_empty() noexcept;

public:
    explicit MetalArgumentBufferPool(id<MTLDevice> device) noexcept;
    [[nodiscard]] MetalArgumentBuffer allocate() noexcept;
    void recycle(MetalArgumentBuffer buffer) noexcept;
};

}
