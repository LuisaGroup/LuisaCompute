//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#include <core/concepts.h>
#include <core/stl/optional.h>
#include <runtime/command.h>
#include <api/common.h>
#include <core/stl/functional.h>

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

public:
    struct Commit {
        CommandList &&cmd_list;
    };

private:
    luisa::vector<luisa::unique_ptr<Command>> _commands;
    luisa::fixed_vector<luisa::move_only_function<void()>, 1> _callbacks;
#ifdef LC_ENABLE_API
    // For backends that use C API only
    // DO NOT USE THIS FIELD OTHERWISE
    luisa::optional<LCCommandList> _c_list;
#endif

public:
    CommandList() noexcept = default;
    ~CommandList() noexcept;
    CommandList(CommandList &&) noexcept = default;
    CommandList &operator=(CommandList &&rhs) noexcept = default;
    void reserve(size_t command_size, size_t callback_size) noexcept;
    CommandList &operator<<(luisa::unique_ptr<Command> &&cmd) noexcept;
    CommandList &operator<<(luisa::move_only_function<void()> &&callback) noexcept;
    void clear() noexcept;
    [[nodiscard]] auto commands() const noexcept { return luisa::span<const luisa::unique_ptr<Command>>{_commands}; }
    [[nodiscard]] auto callbacks() const noexcept { return luisa::span<const luisa::move_only_function<void()>>{_callbacks}; }
    [[nodiscard]] auto &&steal_commands() &&noexcept { return std::move(_commands); }
    [[nodiscard]] auto &&steal_callbacks() &&noexcept { return std::move(_callbacks); }
    [[nodiscard]] auto empty() const noexcept { return _commands.empty() && _callbacks.empty(); }
    [[nodiscard]] auto cmd_size() const noexcept { return _commands.size(); }
    [[nodiscard]] auto callback_size() const noexcept { return _callbacks.size(); }
    [[nodiscard]] auto commit() noexcept { return Commit{std::move(*this)}; }
};

}// namespace luisa::compute
