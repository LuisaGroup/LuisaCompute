//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <utility>
#include <runtime/device.h>

namespace luisa::compute {

class Device;

class Stream : public concepts::Noncopyable {

private:
    Device *_device;
    uint64_t _handle;

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

    template<typename Cmd>
    Stream &operator<<(Cmd &&cmd) {
        _device->_dispatch(_handle, std::forward<Cmd>(cmd));
        return *this;
    }
};

}// namespace luisa::compute
