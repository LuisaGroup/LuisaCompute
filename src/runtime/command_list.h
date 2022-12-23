//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#include <runtime/command.h>

namespace luisa::compute {
class CmdDeser;
class CmdSer;
class LC_RUNTIME_API CommandList : concepts::Noncopyable {
    friend class CmdDeser;
    friend class CmdSer;

private:
    luisa::vector<luisa::unique_ptr<Command>> _commands;

private:
    void _recycle() noexcept;

public:
    CommandList() noexcept = default;
    ~CommandList() noexcept;
    CommandList(CommandList &&) noexcept;
    CommandList &operator=(CommandList &&rhs) noexcept;
    void reserve(size_t size) noexcept;
    void append(luisa::unique_ptr<Command> &&cmd) noexcept;
    void clear() noexcept { _commands.clear(); }
    [[nodiscard]] luisa::vector<luisa::unique_ptr<Command>> steal_commands() noexcept;
    [[nodiscard]] auto begin() const noexcept { return _commands.begin(); }
    [[nodiscard]] auto end() const noexcept { return _commands.end(); }
    [[nodiscard]] auto empty() const noexcept { return _commands.empty(); }
    [[nodiscard]] auto size() const noexcept { return _commands.size(); }

    // for debug
    //    [[nodiscard]] nlohmann::json dump_json() const noexcept;
};

}// namespace luisa::compute
