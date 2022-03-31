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
#include <runtime/command_reorder_visitor.h>
#include <runtime/image.h>
#include <runtime/swap_chain.h>
namespace luisa::compute {

class LC_RUNTIME_API Stream final : public Resource {

public:
    struct Synchronize {};
    friend class CommandBuffer;
    struct Present {
        SwapChain const *swap_chain;
        Image<float> const *img;
    };
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
        Delegate &&operator<<(Present) &&noexcept;
    };

private:
    friend class Device;
    void _dispatch(CommandList command_buffer) noexcept;
    void _dispatch(CommandList command_buffer, luisa::move_only_function<void()> &&func) noexcept;
    explicit Stream(Device::Interface *device) noexcept;
    void _synchronize() noexcept;
    luisa::unique_ptr<CommandReorderVisitor> reorder_visitor;

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
    Stream &operator<<(Present p) noexcept;
};

[[nodiscard]] constexpr auto synchronize() { return Stream::Synchronize(); }
[[nodiscard]] constexpr auto present(SwapChain const &swap_chain, Image<float> const &img) noexcept { return Stream::Present{&swap_chain, &img}; }

}// namespace luisa::compute
