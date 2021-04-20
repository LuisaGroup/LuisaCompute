//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <utility>

#include <core/spin_mutex.h>
#include <runtime/device.h>
#include <runtime/event.h>
#include <runtime/command_buffer.h>

namespace luisa::compute {

class Stream {

public:
    class Delegate {

    private:
        Stream *_stream;
        CommandBuffer _command_buffer;

    private:
        void _commit() noexcept;

    public:
        explicit Delegate(Stream *s) noexcept;
        Delegate(Delegate &&) noexcept;
        Delegate(const Delegate &) noexcept = delete;
        Delegate &operator=(Delegate &&) noexcept = delete;
        Delegate &operator=(const Delegate &) noexcept = delete;
        ~Delegate() noexcept;
        Delegate &operator<<(CommandHandle cmd) noexcept;
        Stream &operator<<(Event::Signal signal) noexcept;
        Stream &operator<<(Event::Wait wait) noexcept;
    };

private:
    Device::Interface *_device;
    uint64_t _handle;

private:
    friend class Device;
    void _dispatch(CommandBuffer command_buffer) noexcept;
    
    explicit Stream(Device &device) noexcept
        : _device{device.impl()},
          _handle{device.impl()->create_stream()} {}

public:
    Stream(Stream &&s) noexcept;
    ~Stream() noexcept;
    Stream &operator=(Stream &&rhs) noexcept;
    Stream &operator<<(Event::Signal signal) noexcept;
    Stream &operator<<(Event::Wait wait) noexcept;
    Delegate operator<<(CommandHandle cmd) noexcept;
    void synchronize() noexcept;
};

}// namespace luisa::compute
