//
// Created by Mike Smith on 2021/3/26.
//

#pragma once

#import <vector>

#import <core/spin_mutex.h>
#import <backends/metal/metal_buffer_view.h>

namespace luisa::compute::metal {

class MetalArgumentBufferPool {

public:
    static constexpr auto argument_buffer_size = 4096u;
    static constexpr auto trunk_size = 64u;

private:
    __weak id<MTLDevice> _device;
    std::vector<MetalBufferView> _available_buffers;
    spin_mutex _mutex;

private:
    void _create_new_trunk_if_empty() noexcept;

public:
    explicit MetalArgumentBufferPool(id<MTLDevice> device) noexcept;
    [[nodiscard]] MetalBufferView allocate() noexcept;
    void recycle(MetalBufferView buffer) noexcept;
};

}
