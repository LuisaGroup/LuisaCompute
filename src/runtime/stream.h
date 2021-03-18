//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <utility>

#include <core/spin_mutex.h>
#include <runtime/device.h>
#include <runtime/command_group.h>

namespace luisa::compute {

class Device;

namespace detail {
struct StreamSyncToken {};
}// namespace detail

[[nodiscard]] constexpr auto synchronize() noexcept { return detail::StreamSyncToken{}; }

class Stream : public concepts::Noncopyable {

private:
    Device *_device;
    uint64_t _handle;
    spin_mutex _mutex;
    

private:
    friend class Device;
    Stream(Device *device, uint64_t handle) noexcept
        : _device{device}, _handle{handle} {}

public:
    Stream(Stream &&s) noexcept
        : _device{s._device},
          _handle{s._handle} {
        s._device = nullptr;
    }

    ~Stream() noexcept {
        if (_device != nullptr) {
            _device->_dispose_stream(_handle);
        }
    }

    Stream &operator=(Stream &&rhs) noexcept {
        _device = rhs._device;
        _handle = rhs._handle;
        rhs._device = nullptr;
        return *this;
    }

    Stream &operator<<(std::unique_ptr<Command> cmd) {
//        _device->_dispatch(_handle, *cmd);
//        return *this;
    }

    Stream &operator<<(std::function<void()> f) {
        _device->_dispatch(_handle, std::move(f));
        return *this;
    }
    
    Stream &operator<<(detail::StreamSyncToken) {
        _device->_synchronize_stream(_handle);
        return *this;
    }

    template<typename Cmd>
    Stream &operator<<(Cmd &&cmd) {
        _device->_dispatch(_handle, *cmd);
        return *this;
    }
};

}// namespace luisa::compute
