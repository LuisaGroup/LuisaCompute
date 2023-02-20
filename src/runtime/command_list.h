//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#include <core/concepts.h>
#include <core/stl/optional.h>
#include <runtime/command.h>
#include <api/common.h>

namespace luisa::compute {

class CmdDeser;
class CmdSer;

namespace detail {
class CommandListConverter;
}

class LC_RUNTIME_API CommandList : concepts::Noncopyable {

    friend class CmdDeser;
    friend class CmdSer;
    friend class detail::CommandListConverter;

private:
    luisa::vector<luisa::unique_ptr<Command>> _commands;

    // For backends that use C API only
    // DO NOT USE THIS FIELD OTHERWISE
    luisa::optional<LCCommandList> _c_list;

public:
    CommandList() noexcept = default;
    ~CommandList() noexcept = default;
    CommandList(CommandList &&) noexcept = default;
    CommandList &operator=(CommandList &&rhs) noexcept = default;
    void reserve(size_t size) noexcept;
    void append(luisa::unique_ptr<Command> &&cmd) noexcept;
    void clear() noexcept { _commands.clear(); }
    [[nodiscard]] luisa::vector<luisa::unique_ptr<Command>> steal_commands() noexcept;
    [[nodiscard]] auto begin() const noexcept { return _commands.begin(); }
    [[nodiscard]] auto end() const noexcept { return _commands.end(); }
    [[nodiscard]] auto empty() const noexcept { return _commands.empty(); }
    [[nodiscard]] auto size() const noexcept { return _commands.size(); }
};

}// namespace luisa::compute
