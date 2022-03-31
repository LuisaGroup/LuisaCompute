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
    CommandBuffer(CommandBuffer &&another) noexcept;
    explicit CommandBuffer(Stream *stream) noexcept;

public:
    ~CommandBuffer() noexcept;
    CommandBuffer &operator=(CommandBuffer &&) noexcept = delete;
    CommandBuffer &operator<<(Command *cmd) &noexcept;
    CommandBuffer &operator<<(Event::Signal) &noexcept;
    CommandBuffer &operator<<(Event::Wait) &noexcept;
    CommandBuffer &operator<<(SwapChain::Present p) &noexcept;
    CommandBuffer &operator<<(Commit) &noexcept;
    CommandBuffer &operator<<(Synchronize) &noexcept;
    void commit() &noexcept { _commit(); }
    void commit(luisa::move_only_function<void()> func) &noexcept;
    void synchronize() &noexcept;
    [[nodiscard]] auto &stream() noexcept { return *_stream; }
};

[[nodiscard]] constexpr auto commit() noexcept { return CommandBuffer::Commit{}; }
[[nodiscard]] constexpr auto synchronize() noexcept { return CommandBuffer::Synchronize{}; }

}// namespace luisa::compute
