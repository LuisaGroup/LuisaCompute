//
// Created by Mike Smith on 2021/7/20.
//

#pragma once

#include <runtime/event.h>
#include <runtime/command_list.h>
#include <runtime/swap_chain.h>

namespace luisa::compute {

class Command;
class Stream;

class LC_RUNTIME_API CommandBuffer {

public:
    struct Commit {};
    struct Synchronize {};

private:
    Stream *_stream;
    CommandList _command_list;

private:
    friend class Stream;
    void _commit() &noexcept;
    explicit CommandBuffer(Stream *stream) noexcept;

public:
    ~CommandBuffer() noexcept;
    CommandBuffer(CommandBuffer const &another) = delete;
    CommandBuffer(CommandBuffer &&another) noexcept;
    CommandBuffer &operator=(CommandBuffer &&) noexcept = delete;
    CommandBuffer &operator<<(luisa::unique_ptr<Command> &&cmd) &noexcept;
    CommandBuffer &operator<<(Event::Signal) &noexcept;
    CommandBuffer &operator<<(Event::Wait) &noexcept;
    CommandBuffer &operator<<(SwapChain::Present p) &noexcept;
    CommandBuffer &operator<<(Commit) &noexcept;
    CommandBuffer &operator<<(Synchronize) &noexcept;
    CommandBuffer &operator<<(luisa::move_only_function<void()> &&f) &noexcept;

    auto empty() const noexcept { return _command_list.empty(); }
    void commit() &noexcept { _commit(); }
    void synchronize() &noexcept;
    [[nodiscard]] auto &stream() noexcept { return *_stream; }

    // compound commands
    template<typename... T>
    decltype(auto) operator<<(std::tuple<T...> args) &noexcept {
        auto encode = [this]<size_t... i>(std::tuple<T...> a, std::index_sequence<i...>) noexcept -> decltype(auto) {
            return (*this << ... << std::move(std::get<i>(a)));
        };
        return encode(std::move(args), std::index_sequence_for<T...>{});
    }
};
// Commmit all commands to the parent stream
[[nodiscard]] constexpr auto commit() noexcept { return CommandBuffer::Commit{}; }
// Commit all commands to the parent stream and wait until everything is finished
[[nodiscard]] constexpr auto synchronize() noexcept { return CommandBuffer::Synchronize{}; }

}// namespace luisa::compute
