#pragma once

#include <atomic>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/spin_mutex.h>
#include "metal_api.h"

namespace luisa::compute::metal {

struct MetalCallbackContext;

class MetalEvent {

private:
    MTL::SharedEvent *_handle;
    luisa::shared_ptr<std::atomic_uint64_t> _host_completed_value;

public:
    explicit MetalEvent(MTL::Device *device) noexcept;
    ~MetalEvent() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] bool is_completed(uint64_t value) const noexcept;
    void signal(MTL::CommandBuffer *command_buffer, uint64_t value) noexcept;
    [[nodiscard]] MetalCallbackContext *host_signal_callback(
        uint64_t value) const noexcept;
    void wait(MTL::CommandBuffer *command_buffer, uint64_t value) noexcept;
    void synchronize(uint64_t value) noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal
