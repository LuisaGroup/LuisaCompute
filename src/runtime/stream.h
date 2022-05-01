//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include "core/stl.h"
#include <utility>

#include <core/spin_mutex.h>
#include <runtime/resource.h>
#include <runtime/event.h>
#include <runtime/command_list.h>
#include <runtime/command_buffer.h>
#include <runtime/command_reorder_visitor.h>
#include <runtime/image.h>
#include <runtime/swap_chain.h>
#include <runtime/command_scheduler.h>

namespace luisa::compute {

class LC_RUNTIME_API Stream final : public Resource {

public:
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
        Delegate &&operator<<(luisa::move_only_function<void()> &&f) &&noexcept;
        Delegate &&operator<<(CommandBuffer::Commit) &&noexcept;
        Delegate &&operator<<(CommandBuffer::Synchronize) &&noexcept;
        Delegate &&operator<<(SwapChain::Present present) &&noexcept;

        // compound commands
        template<typename... T>
        decltype(auto) operator<<(std::tuple<T...> args) noexcept {
            auto encode = [this]<size_t... i>(std::tuple<T...> a, std::index_sequence<i...>) noexcept {
                return (std::move(*this) << ... << std::move(std::get<i>(a)));
            };
            return encode(std::move(args), std::index_sequence_for<T...>{});
        }
    };

private:
    luisa::unique_ptr<CommandScheduler> _scheduler;
    friend class Device;
    void _dispatch(CommandList command_buffer) noexcept;
    explicit Stream(Device::Interface *device, bool for_present = false) noexcept;
    void _synchronize() noexcept;
    luisa::unique_ptr<CommandReorderVisitor> reorder_visitor;

public:
    Stream() noexcept = default;
    using Resource::operator bool;
    Stream &operator<<(Event::Signal signal) noexcept;
    Stream &operator<<(Event::Wait wait) noexcept;
    Stream &operator<<(CommandBuffer::Synchronize) noexcept;
    Stream &operator<<(CommandBuffer::Commit) noexcept { return *this; }
    Stream &operator<<(luisa::move_only_function<void()> &&f) noexcept;
    Delegate operator<<(Command *cmd) noexcept;
    [[nodiscard]] auto command_buffer() noexcept { return CommandBuffer{this}; }
    [[nodiscard]] auto native_handle() const noexcept { return device()->stream_native_handle(handle()); }
    void synchronize() noexcept { _synchronize(); }
    Stream &operator<<(SwapChain::Present p) noexcept;

    // compound commands
    template<typename... T>
    decltype(auto) operator<<(std::tuple<T...> args) noexcept {
        auto encode = [this]<size_t... i>(std::tuple<T...> a, std::index_sequence<i...>) noexcept {
            return (*this << ... << std::move(std::get<i>(a)));
        };
        return encode(std::move(args), std::index_sequence_for<T...>{});
    }
};

}// namespace luisa::compute
