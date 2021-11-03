//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <utility>

#include <core/spin_mutex.h>
#include <runtime/resource.h>
#include <runtime/event.h>
#include <runtime/command_list.h>
#include <runtime/command_buffer.h>

namespace luisa::compute {

class Stream final : public Resource {

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
        Delegate &&operator<<(CommandBuffer::Commit) &&noexcept;
        Delegate &&operator<<(Synchronize) &&noexcept;
    };

private:
    friend class Device;
    void _dispatch(CommandList command_buffer) noexcept;
    explicit Stream(Device::Interface *device) noexcept;
    void _synchronize() noexcept;

public:
    Stream() noexcept = default;
    using Resource::operator bool;
    Stream &operator<<(Event::Signal signal) noexcept;
    Stream &operator<<(Event::Wait wait) noexcept;
    Stream &operator<<(Synchronize) noexcept;
    Stream &operator<<(CommandBuffer::Commit) noexcept { return *this; }
    Delegate operator<<(Command *cmd) noexcept;
    [[nodiscard]] auto command_buffer() noexcept { return CommandBuffer{this}; }
    [[nodiscard]] auto native_handle() const noexcept { return device()->stream_native_handle(handle()); }
    void synchronize() noexcept { _synchronize(); }
};

[[nodiscard]] constexpr auto synchronize() noexcept { return Stream::Synchronize{}; }

}// namespace luisa::compute
