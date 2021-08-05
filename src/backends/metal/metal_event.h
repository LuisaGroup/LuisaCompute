//
// Created by Mike Smith on 2021/4/8.
//

#pragma once

#import <Metal/Metal.h>
#import <util/spin_mutex.h>

namespace luisa::compute::metal {

class MetalEvent {

private:
    id<MTLEvent> _handle;
    __weak id<MTLCommandBuffer> _last{nullptr};
    uint64_t _counter{0u};
    spin_mutex _mutex;

public:
    explicit MetalEvent(id<MTLEvent> handle) noexcept
        : _handle{handle} {}
    ~MetalEvent() noexcept { _handle = nullptr; }

    void signal(id<MTLCommandBuffer> command_buffer) noexcept {
        auto value = [this, command_buffer] {
            std::scoped_lock lock{_mutex};
            _last = command_buffer;
            return ++_counter;
        }();
        [command_buffer encodeSignalEvent:_handle
                                    value:value];
    }

    void wait(id<MTLCommandBuffer> command_buffer) noexcept {
        if (auto value = [this] {
                std::scoped_lock lock{_mutex};
                return _counter;
            }();
            value != 0u) {
            [command_buffer encodeWaitForEvent:_handle
                                         value:value];
        }
    }

    void synchronize() noexcept {
        if (auto last = [this]() noexcept
            -> id<MTLCommandBuffer> {
                std::scoped_lock lock{_mutex};
                auto cmd = _last;
                _last = nullptr;
                return cmd;
            }()) [[likely]] { [last waitUntilCompleted]; }
    }
};

}// namespace luisa::compute::metal
