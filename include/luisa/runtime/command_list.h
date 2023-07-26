#pragma once

#include <luisa/core/concepts.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/functional.h>
#include <luisa/runtime/rhi/command.h>

#ifdef LUISA_ENABLE_API
#include <luisa/api/common.h>
#endif
namespace lc::validation {
class Device;
}// namespace lc::validation
namespace luisa::compute {

class LC_RUNTIME_API CommandList : concepts::Noncopyable {
    friend class lc::validation::Device;

public:
    class Commit;
    using CommandContainer = luisa::vector<luisa::unique_ptr<Command>>;
    using CallbackContainer = luisa::vector<luisa::move_only_function<void()>>;

private:
    CommandContainer _commands;
    CallbackContainer _callbacks;
    bool _committed{false};

public:
    CommandList() noexcept = default;
    ~CommandList() noexcept;
    CommandList(CommandList &&another) noexcept;
    CommandList &operator=(CommandList &&rhs) noexcept = delete;
    [[nodiscard]] static CommandList create(size_t reserved_command_size = 0u,
                                            size_t reserved_callback_size = 0u) noexcept;

    void reserve(size_t command_size, size_t callback_size) noexcept;
    CommandList &operator<<(luisa::unique_ptr<Command> &&cmd) noexcept;
    CommandList &append(luisa::unique_ptr<Command> &&cmd) noexcept;
    CommandList &add_callback(luisa::move_only_function<void()> &&callback) noexcept;
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
    friend class CommandList;
    explicit Commit(CommandList &&list) noexcept
        : _list{std::move(list)} {}
    Commit(Commit &&) noexcept = default;

public:
    Commit &operator=(Commit &&) noexcept = delete;
    Commit &operator=(const Commit &) noexcept = delete;
    [[nodiscard]] auto command_list() && noexcept { return std::move(_list); }
};

}// namespace luisa::compute
