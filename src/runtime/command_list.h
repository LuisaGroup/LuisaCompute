//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#include <core/stl.h>
#include <runtime/command.h>

namespace luisa::compute {

class LC_RUNTIME_API CommandList : concepts::Noncopyable {

private:
    luisa::vector<Command *> _commands;

private:
    void _recycle() noexcept;

public:
    CommandList() noexcept = default;
    ~CommandList() noexcept;
    CommandList(CommandList &&) noexcept;
    CommandList &operator=(CommandList &&rhs) noexcept;
    void append(Command *cmd) noexcept;
    void clear() noexcept { _commands.clear(); }
    [[nodiscard]] luisa::vector<Command *> steal_commands() noexcept;
    [[nodiscard]] auto begin() const noexcept { return _commands.begin(); }
    [[nodiscard]] auto end() const noexcept { return _commands.end(); }
    [[nodiscard]] auto empty() const noexcept { return _commands.empty(); }
    [[nodiscard]] auto size() const noexcept { return _commands.size(); }
};

}// namespace luisa::compute
