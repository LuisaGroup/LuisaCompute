//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <utility>

#include <core/spin_mutex.h>
#include <runtime/device.h>
#include <runtime/event.h>
#include <runtime/command_list.h>
#include <runtime/command_buffer.h>

namespace luisa::compute {

class Stream {

public:
    struct Synchronize {};
    friend class CommandBuffer;

    class Delegate {

    private:
        Stream *_stream;
        CommandList _command_list;

    private:
        void _commit() noexcept;

    public:
        explicit Delegate(Stream *s) noexcept;
        ~Delegate() noexcept;
        Delegate(Delegate &&) noexcept;
        Delegate(const Delegate &) noexcept = delete;
        Delegate &&operator=(Delegate &&) noexcept = delete;
        Delegate &&operator=(const Delegate &) noexcept = delete;
        Delegate &&operator<<(Command *cmd) &&noexcept;
        Delegate &&operator<<(Event::Signal signal) &&noexcept;
        Delegate &&operator<<(Event::Wait wait) &&noexcept;
        Delegate &&operator<<(Synchronize) &&noexcept;
    };

private:
    Device::Handle _device;
    uint64_t _handle{};

private:
    friend class Device;
    void _dispatch(CommandList command_buffer) noexcept;

    explicit Stream(Device::Handle device) noexcept
        : _device{std::move(device)},
          _handle{_device->create_stream()} {}
    void _synchronize() noexcept;
    void _destroy() noexcept;

public:
    Stream() noexcept = default;
    Stream(Stream &&s) noexcept = default;
    ~Stream() noexcept;
    Stream &operator=(Stream &&rhs) noexcept;
    Stream &operator<<(Event::Signal signal) noexcept;
    Stream &operator<<(Event::Wait wait) noexcept;
    Stream &operator<<(Synchronize) noexcept;
    void synchronize() noexcept { _synchronize(); }
    Delegate operator<<(Command *cmd) noexcept;
    [[nodiscard]] auto command_buffer() noexcept { return CommandBuffer{this}; }
    [[nodiscard]] explicit operator bool() const noexcept { return _device != nullptr; }
};

[[nodiscard]] constexpr auto synchronize() noexcept { return Stream::Synchronize{}; }

}// namespace luisa::compute
