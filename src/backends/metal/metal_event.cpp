#include <mutex>

#include <luisa/core/logging.h>
#include "metal_event.h"

namespace luisa::compute::metal {

MetalEvent::MetalEvent(MTL::Device *device) noexcept
    : _handle{device->newSharedEvent()} {}

MetalEvent::~MetalEvent() noexcept {
    _handle->release();
}

void MetalEvent::signal(MTL::CommandBuffer *command_buffer, uint64_t value) noexcept {
    command_buffer->encodeSignalEvent(_handle, value);
}

bool MetalEvent::is_completed(uint64_t value) const noexcept {
    return _handle->signaledValue() >= value;
}

void MetalEvent::wait(MTL::CommandBuffer *command_buffer, uint64_t value) noexcept {
    if (value == 0u) {// not signaled yet
        LUISA_WARNING_WITH_LOCATION(
            "MetalEvent::wait() is called "
            "before any signal event.");
    } else {// encode a wait event into a new command buffer
        command_buffer->encodeWait(_handle, value);
    }
}

void MetalEvent::synchronize(uint64_t value) noexcept {
    if (value == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "MetalEvent::synchronize() is called "
            "before any signal event.");
        return;
    }
    while (!is_completed(value)) {
        // wait until the signaled value is greater than or equal to the value to wait
        std::this_thread::yield();
    }
}

void MetalEvent::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        _handle->setLabel(nullptr);
    } else {
        auto mtl_name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
        _handle->setLabel(mtl_name);
        mtl_name->release();
    }
}

}// namespace luisa::compute::metal

