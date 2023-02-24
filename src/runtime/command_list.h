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
}// namespace detail

class LC_RUNTIME_API CommandList : concepts::Noncopyable {

    friend class CmdDeser;
    friend class CmdSer;
    friend class detail::CommandListConverter;

public:
    class Commit;
    using CommandContainer = luisa::vector<luisa::unique_ptr<Command>>;
    using CallbackContainer = luisa::vector<luisa::move_only_function<void()>>;

private:
    CommandContainer _commands;
    CallbackContainer _callbacks;
    bool _committed{false};

#ifdef LC_ENABLE_API
    // For backends that use C API only
    // DO NOT USE THIS FIELD OTHERWISE
    luisa::optional<LCCommandList> _c_list;
#endif

public:
    CommandList() noexcept = default;
    ~CommandList() noexcept;
    CommandList(CommandList &&another) noexcept;
    CommandList &operator=(CommandList &&rhs) noexcept = delete;
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
    [[nodiscard]] CommandContainer steal_commands() noexcept;
    [[nodiscard]] CallbackContainer steal_callbacks() noexcept;
    [[nodiscard]] auto empty() const noexcept { return _commands.empty() && _callbacks.empty(); }
    [[nodiscard]] Commit commit() noexcept;
};

class CommandList::Commit {

private:
    CommandList _list;

private:
    friend class Stream;
    friend class CommandList;
    explicit Commit(CommandList &&list) noexcept
        : _list{std::move(list)} {}
    Commit(Commit &&) noexcept = default;
    Commit &operator=(Commit &&) noexcept = delete;
};

}// namespace luisa::compute
