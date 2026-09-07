#pragma once

#include <luisa/core/concepts.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/functional.h>
#include <luisa/runtime/rhi/command.h>

namespace lc::validation {
class Device;
}// namespace lc::validation

namespace luisa::compute {
struct SwapchainPresent;
class LUISA_RUNTIME_API CommandList : concepts::Noncopyable {
    friend class lc::validation::Device;

public:
    class Commit;
    using CommandContainer = luisa::vector<luisa::unique_ptr<Command>>;
    using CallbackContainer = luisa::vector<luisa::move_only_function<void()>>;
    using PresentContainer = luisa::vector<SwapchainPresent>;
    using DtorCallbacks = luisa::fixed_vector<luisa::move_only_function<void()>, 1>;

private:
    CommandContainer _commands;
    CallbackContainer _callbacks;
    PresentContainer _presents;
    DtorCallbacks _dtor_callbacks;
    bool _committed{false};

public:
    CommandList() noexcept;
    CommandList(
        CommandContainer &&commands,
        CallbackContainer &&callbacks,
        PresentContainer &&presents,
        DtorCallbacks &&dtor_callbacks) noexcept;
    ~CommandList() noexcept;
    CommandList(CommandList &&another) noexcept;
    CommandList &operator=(CommandList &&rhs) noexcept = delete;
    [[nodiscard]] static CommandList create(size_t reserved_command_size = 0u,
                                            size_t reserved_callback_size = 0u) noexcept;

    void reserve(size_t command_size, size_t callback_size, size_t present_size = 1) noexcept;
    CommandList &operator<<(luisa::unique_ptr<Command> &&cmd) noexcept;
    CommandList &append(luisa::unique_ptr<Command> &&cmd) noexcept;
    CommandList &add_callback(luisa::move_only_function<void()> &&callback) noexcept;
    // Run after command-list destruct (usually after commit)
    CommandList &add_dtor_callback(luisa::move_only_function<void()> &&callback) noexcept;
    CommandList &add_present(SwapchainPresent &&present) noexcept;

    CommandList &add_range(CommandList &&cmdlist) noexcept;
    CommandList &operator<<(CommandList &&cmdlist) noexcept {
        return add_range(std::move(cmdlist));
    }
    void clear() noexcept;
    [[nodiscard]] auto commands() const noexcept { return luisa::span{_commands}; }
    [[nodiscard]] auto callbacks() const noexcept { return luisa::span{_callbacks}; }
    [[nodiscard]] luisa::span<const SwapchainPresent> presents() const noexcept;
    [[nodiscard]] CommandContainer steal_commands() noexcept;
    [[nodiscard]] CallbackContainer steal_callbacks() noexcept;
    [[nodiscard]] PresentContainer steal_presents() noexcept;
    [[nodiscard]] bool empty() const noexcept;
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
