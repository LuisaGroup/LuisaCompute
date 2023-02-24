//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <utility>

#include <core/spin_mutex.h>
#include <runtime/resource.h>
#include <runtime/event.h>
#include <runtime/command_list.h>
#include <runtime/swap_chain.h>
#include <runtime/stream_tag.h>

namespace luisa::compute {

class LC_RUNTIME_API Stream final : public Resource {

public:
    struct Commit {};
    struct Synchronize {};

    class LC_RUNTIME_API Delegate {

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
        Delegate &&operator<<(luisa::unique_ptr<Command> &&cmd) &&noexcept;
        Delegate &&operator<<(luisa::move_only_function<void()> &&f) &&noexcept;
        Stream &operator<<(Event::Signal &&signal) &&noexcept;
        Stream &operator<<(Event::Wait &&wait) &&noexcept;
        Stream &operator<<(SwapChain::Present &&present) &&noexcept;
        Stream &operator<<(CommandList::Commit &&commit) &&noexcept;
        Stream &operator<<(Synchronize &&) &&noexcept;
        Stream &operator<<(Commit &&) &&noexcept;

        // compound commands
        template<typename... T>
        decltype(auto) operator<<(std::tuple<T...> args) &&noexcept {
            auto encode = [this]<size_t... i>(std::tuple<T...> a, std::index_sequence<i...>) noexcept -> decltype(auto) {
                return (std::move(*this) << ... << std::move(std::get<i>(a)));
            };
            return encode(std::move(args), std::index_sequence_for<T...>{});
        }
    };

private:
    friend class Device;
    StreamTag _stream_tag;
    void _dispatch(CommandList &&command_buffer) noexcept;
    explicit Stream(DeviceInterface *device, StreamTag stream_tag) noexcept;
    void _synchronize() noexcept;

public:
    Stream() noexcept = default;
    Stream(Stream &&) noexcept = default;
    Stream(Stream const &) noexcept = delete;
    Stream &operator=(Stream &&) noexcept = default;
    Stream &operator=(Stream const &) noexcept = delete;
    using Resource::operator bool;
    Delegate operator<<(luisa::unique_ptr<Command> &&cmd) noexcept;
    Delegate operator<<(luisa::move_only_function<void()> &&f) noexcept;
    Stream &operator<<(Event::Signal &&signal) noexcept;
    Stream &operator<<(Event::Wait &&wait) noexcept;
    Stream &operator<<(CommandList::Commit &&commit) noexcept;
    Stream &operator<<(SwapChain::Present &&p) noexcept;
    Stream &operator<<(Synchronize &&) noexcept;
    void synchronize() noexcept { _synchronize(); }
    [[nodiscard]] auto stream_tag() const noexcept { return _stream_tag; }

    // compound commands
    template<typename... T>
    decltype(auto) operator<<(std::tuple<T...> &&args) noexcept {
        return Delegate{this} << std::move(args);
    }
};

[[nodiscard]] constexpr auto commit() noexcept { return Stream::Commit{}; }
[[nodiscard]] constexpr auto synchronize() noexcept { return Stream::Synchronize{}; }

}// namespace luisa::compute
