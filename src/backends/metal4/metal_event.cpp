#include <mutex>

#include <luisa/core/logging.h>
#include "metal_callback_context.h"
#include "metal_event.h"

namespace luisa::compute::metal {

MetalEvent::MetalEvent(MTL::Device *device) noexcept
    : _handle{device->newSharedEvent()},
      _host_completed_value{
          luisa::make_shared<std::atomic_uint64_t>(0u)} {}

MetalEvent::~MetalEvent() noexcept {
    _handle->release();
}

MetalCallbackContext *MetalEvent::host_signal_callback(
    uint64_t value) const noexcept {
    return FunctionCallbackContext::create(
        [completed = _host_completed_value, value]() noexcept {
            auto current = completed->load(std::memory_order_relaxed);
            while (current < value &&
                   !completed->compare_exchange_weak(
                       current, value, std::memory_order_release,
                       std::memory_order_relaxed)) {}
        });
}

bool MetalEvent::is_completed(uint64_t value) const noexcept {
    return _handle->signaledValue() >= value &&
           _host_completed_value->load(std::memory_order_acquire) >= value;
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
