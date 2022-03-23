//
// Created by Mike Smith on 2021/7/20.
//

#pragma once

#include <runtime/event.h>
#include <runtime/command_list.h>

namespace luisa::compute {

class Command;
class Stream;

class LC_RUNTIME_API CommandBuffer {

public:
    struct Commit {};

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
    CommandBuffer &operator<<(Commit) &noexcept;
    void commit() &noexcept { _commit(); }
    void commit(luisa::move_only_function<void()> &&func) &noexcept;
    [[nodiscard]] auto &stream() noexcept { return *_stream; }
};

[[nodiscard]] constexpr auto commit() noexcept { return CommandBuffer::Commit{}; }

}// namespace luisa::compute
