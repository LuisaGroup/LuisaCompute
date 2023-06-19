//
// Created by Mike Smith on 2023/4/15.
//

#include <mutex>

#include <luisa/core/logging.h>
#include "metal_event.h"

namespace luisa::compute::metal {

MetalEvent::MetalEvent(MTL::Device *device) noexcept
    : _handle{device->newSharedEvent()} {}

MetalEvent::~MetalEvent() noexcept {
    _handle->release();
}

void MetalEvent::signal(MTL::CommandBuffer *command_buffer) noexcept {
    // encode a signal event into a new command buffer
    auto value = [this] {
        std::scoped_lock lock{_mutex};
        return ++_signaled_value;
    }();
    command_buffer->encodeSignalEvent(_handle, value);
}

uint64_t MetalEvent::value_to_wait() const noexcept {
    std::scoped_lock lock{_mutex};
    return _signaled_value;
}

bool MetalEvent::is_completed() const noexcept {
    return _handle->signaledValue() >= value_to_wait();
}

void MetalEvent::wait(MTL::CommandBuffer *command_buffer) noexcept {
    auto value = value_to_wait();
    if (value == 0u) {// not signaled yet
        LUISA_WARNING_WITH_LOCATION(
            "MetalEvent::wait() is called "
            "before any signal event.");
    } else {// encode a wait event into a new command buffer
        command_buffer->encodeWait(_handle, value);
    }
}

void MetalEvent::synchronize() noexcept {
    auto value = value_to_wait();
    if (value == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "MetalEvent::synchronize() is called "
            "before any signal event.");
        return;
    }
    while (_handle->signaledValue() < value) {
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

