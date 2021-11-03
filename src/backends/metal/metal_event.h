//
// Created by Mike Smith on 2021/4/8.
//

#pragma once

#include <core/allocator.h>

#import <Metal/Metal.h>
#import <core/spin_mutex.h>

namespace luisa::compute::metal {

class MetalEvent {

private:
    id<MTLEvent> _handle;
    std::atomic<uint64_t> _counter{0u};
    spin_mutex _mutex;
    __weak id<MTLCommandBuffer> _observer;

public:
    explicit MetalEvent(id<MTLEvent> handle) noexcept
        : _handle{handle} {}
    ~MetalEvent() noexcept { _handle = nullptr; }

    void signal(id<MTLCommandBuffer> command_buffer) noexcept {
        auto value = _counter.fetch_add(1u, std::memory_order::release) + 1u;
        [command_buffer encodeSignalEvent:_handle
                                    value:value];
        std::scoped_lock lock{_mutex};
        _observer = command_buffer;
    }

    void wait(id<MTLCommandBuffer> command_buffer) noexcept {
        if (auto value = _counter.load(std::memory_order::acquire);
            value == 0u) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Ignoring MetalEvent::wait() without signaling.");
        } else [[likely]] {
            [command_buffer encodeWaitForEvent:_handle
                                         value:value];
        }
    }

    void synchronize() noexcept {
        if (auto observer = [this] {
                std::scoped_lock lock{_mutex};
                return (id<MTLCommandBuffer>)_observer;
            }()) { [observer waitUntilCompleted]; }
    }
};

}// namespace luisa::compute::metal
