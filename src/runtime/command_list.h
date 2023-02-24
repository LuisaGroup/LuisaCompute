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
    class Commit {
        friend class Stream;
        friend class CommandList;
        CommandList &&_cmd_list;
        Commit(CommandList &&cmd_list) : _cmd_list{std::move(cmd_list)} {}
        Commit(const Commit &) noexcept = delete;
        Commit(Commit &&rhs) noexcept = default;
        Commit &operator=(Commit &&) noexcept = delete;
        Commit &operator=(const Commit &) noexcept = delete;

    public:
        ~Commit() noexcept = default;
    };

    using CommandContainer = luisa::vector<luisa::unique_ptr<Command>>;
    using CallbackContainer = luisa::fixed_vector<luisa::move_only_function<void()>, 1>;

private:
    CommandContainer _commands;
    CallbackContainer _callbacks;
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
    [[nodiscard]] static CommandList create(size_t reserved_command_size = 0u,
                                            size_t reserved_callback_size = 0u) noexcept;

    void reserve(size_t command_size, size_t callback_size) noexcept;
    CommandList &operator<<(luisa::unique_ptr<Command> &&cmd) noexcept;
    CommandList &operator<<(luisa::move_only_function<void()> &&callback) noexcept;
    CommandList &append(luisa::unique_ptr<Command> &&cmd) noexcept;
    CommandList &append(luisa::move_only_function<void()> &&callback) noexcept;
    void clear() noexcept;
    [[nodiscard]] auto commands() const noexcept { return luisa::span{_commands}; }
    [[nodiscard]] auto callbacks() const noexcept { return luisa::span{_callbacks}; }
    [[nodiscard]] CallbackContainer steal_callbacks() &&noexcept;
    [[nodiscard]] auto empty() const noexcept { return _commands.empty() && _callbacks.empty(); }
    [[nodiscard]] auto commit() noexcept { return Commit{std::move(*this)}; }
};

}// namespace luisa::compute
