//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <core/stl/string.h>
#include <core/spin_mutex.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalEvent {

private:
    MTL::Event *_handle;
    uint64_t _signaled_value{0u};
    MTL::CommandBuffer *_signaled_buffer{nullptr};
    spin_mutex _mutex;

public:
    explicit MetalEvent(MTL::Device *device) noexcept;
    ~MetalEvent() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void signal(MTL::CommandQueue *queue) noexcept;
    void wait(MTL::CommandQueue *queue) noexcept;
    void synchronize() noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal
