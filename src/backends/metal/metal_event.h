#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/core/spin_mutex.h>
#include "metal_api.h"

namespace luisa::compute::metal {

class MetalEvent {

private:
    MTL::SharedEvent *_handle;

public:
    explicit MetalEvent(MTL::Device *device) noexcept;
    ~MetalEvent() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] bool is_completed(uint64_t value) const noexcept;
    void signal(MTL::CommandBuffer *command_buffer, uint64_t value) noexcept;
    void wait(MTL::CommandBuffer *command_buffer, uint64_t value) noexcept;
    void synchronize(uint64_t value) noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal

