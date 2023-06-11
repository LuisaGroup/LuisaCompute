//
// Created by Mike Smith on 2023/4/15.
//

#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/core/spin_mutex.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalEvent {

private:
    MTL::SharedEvent *_handle;
    uint64_t _signaled_value{0u};
    mutable spin_mutex _mutex;

public:
    explicit MetalEvent(MTL::Device *device) noexcept;
    ~MetalEvent() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] uint64_t value_to_wait() const noexcept;
    void signal(MTL::CommandBuffer *command_buffer) noexcept;
    void wait(MTL::CommandBuffer *command_buffer) noexcept;
    void synchronize() noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal

