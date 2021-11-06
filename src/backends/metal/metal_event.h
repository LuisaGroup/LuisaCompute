//
// Created by Mike Smith on 2021/4/8.
//

#pragma once

#include <core/allocator.h>

#import <Metal/Metal.h>
#import <core/spin_mutex.h>

namespace luisa::compute::metal {

class MetalEvent {

public:
    struct Signaler {
        __weak id<MTLCommandBuffer> handle;
    };

private:
    id<MTLEvent> _handle;
    uint64_t _counter{0u};
    std::vector<Signaler> _signalers;

private:
    void _purge() noexcept {
        _signalers.erase(
            std::remove_if(
                _signalers.begin(),
                _signalers.end(),
                [](auto &&s) noexcept {
                    return s.handle == nullptr;
                }),
            _signalers.end());
    }

public:
    explicit MetalEvent(id<MTLEvent> handle) noexcept
        : _handle{handle} {}
    ~MetalEvent() noexcept { _handle = nullptr; }

    void signal(id<MTLCommandBuffer> command_buffer) noexcept {
        [command_buffer encodeSignalEvent:_handle
                                    value:++_counter];
        _purge();
        _signalers.emplace_back(Signaler{command_buffer});
    }

    void wait(id<MTLCommandBuffer> command_buffer) noexcept {
        if (_counter == 0u) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Ignoring MetalEvent::wait() without signaling.");
        } else [[likely]] {
            [command_buffer encodeWaitForEvent:_handle
                                         value:_counter];
        }
    }

    void synchronize() noexcept {
        for (auto &&s : _signalers) {
            if (id<MTLCommandBuffer> h = s.handle) {
                [h waitUntilCompleted];
            }
        }
        _signalers.clear();
    }
};

}// namespace luisa::compute::metal
