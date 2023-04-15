//
// Created by Mike Smith on 2023/4/15.
//

#include <mutex>

#include <core/logging.h>
#include <backends/metal/metal_event.h>

namespace luisa::compute::metal {

MetalEvent::MetalEvent(MTL::Device *device) noexcept
    : _handle{device->newEvent()} {}

MetalEvent::~MetalEvent() noexcept {
    synchronize();
    _handle->release();
}

void MetalEvent::signal(MTL::CommandQueue *queue) noexcept {

    // encode a signal event into a new command buffer
    auto autorelease_pool = NS::AutoreleasePool::alloc()->init();
    auto [command_buffer, signaled_value, old_command_buffer] = [this, queue] {
        auto buffer = queue->commandBufferWithUnretainedReferences();
        std::scoped_lock lock{_mutex};
        auto old_buffer = _signaled_buffer;
        _signaled_buffer = buffer;
        auto value = ++_signaled_value;
        return std::make_tuple(buffer, value, old_buffer);
    }();
    if (old_command_buffer) { old_command_buffer->release(); }
    command_buffer->retain();
    command_buffer->encodeSignalEvent(_handle, signaled_value);
    command_buffer->commit();
    autorelease_pool->release();
}

void MetalEvent::wait(MTL::CommandQueue *queue) noexcept {

    auto signaled_value = [this] {
        std::scoped_lock lock{_mutex};
        return _signaled_value;
    }();

    if (signaled_value == 0u) {// not signaled yet
        LUISA_WARNING_WITH_LOCATION(
            "MetalEvent::wait() is called "
            "before any signal event.");
    } else {// encode a wait event into a new command buffer
        auto autorelease_pool = NS::AutoreleasePool::alloc()->init();
        auto command_buffer = queue->commandBufferWithUnretainedReferences();
        command_buffer->encodeWait(_handle, signaled_value);
        command_buffer->commit();
        autorelease_pool->release();
    }
}

void MetalEvent::synchronize() noexcept {

    auto signaled_buffer = [this] {
        std::scoped_lock lock{_mutex};
        return _signaled_buffer ?
                   _signaled_buffer->retain() :
                   nullptr;
    }();

    if (signaled_buffer) {
        // wait until the signaled buffer is completed
        signaled_buffer->waitUntilCompleted();
        // release the signaled buffer
        signaled_buffer->release();
        // if the signaled buffer is still the same,
        // reset it, so we do not need to wait again
        auto still_current = [this, s = signaled_buffer] {
            std::scoped_lock lock{_mutex};
            if (_signaled_buffer != s) { return false; }
            _signaled_buffer = nullptr;
            return true;
        }();
        if (still_current) { signaled_buffer->release(); }
    }
}

void MetalEvent::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        _handle->setLabel(nullptr);
    } else {
        luisa::string mtl_name{name};
        auto autorelease_pool = NS::AutoreleasePool::alloc()->init();
        _handle->setLabel(NS::String::string(mtl_name.c_str(), NS::UTF8StringEncoding));
        autorelease_pool->release();
    }
}

}// namespace luisa::compute::metal
