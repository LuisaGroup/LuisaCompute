#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

CommandList::~CommandList() noexcept {
    LUISA_ASSERT(_committed || empty(),
                 "Destructing non-empty command list. "
                 "Did you forget to commit?");
    for (auto &i : _dtor_callbacks) {
        i();
    }
}

void CommandList::reserve(size_t command_size, size_t callback_size, size_t present_size) noexcept {
    if (command_size) { _commands.reserve(command_size); }
    if (callback_size) { _callbacks.reserve(callback_size); }
    if (present_size) { _presents.reserve(present_size); }
}

void CommandList::clear() noexcept {
    _commands.clear();
    _callbacks.clear();
    _presents.clear();
    _dtor_callbacks.clear();
    _committed = false;
}

CommandList &CommandList::append(luisa::unique_ptr<Command> &&cmd) noexcept {
    if (cmd) { _commands.emplace_back(std::move(cmd)); }
    return *this;
}

CommandList &CommandList::add_callback(luisa::move_only_function<void()> &&callback) noexcept {
    if (callback) {
        if (_callbacks.empty()) [[likely]] { _callbacks.reserve(2); }
        _callbacks.emplace_back(std::move(callback));
    }
    return *this;
}

CommandList &CommandList::add_range(CommandList &&cmdlist) noexcept {
    if (cmdlist.empty()) [[unlikely]] { return *this; }
    // move commands into this command list
    _commands.reserve(_commands.size() + cmdlist._commands.size());
    for (auto &&cmd : cmdlist._commands) { _commands.emplace_back(std::move(cmd)); }
    // move callbacks into this command list
    _callbacks.reserve(_callbacks.size() + cmdlist._callbacks.size());
    for (auto &&cb : cmdlist._callbacks) { _callbacks.emplace_back(std::move(cb)); }
    // move presents into this command list
    _presents.reserve(_presents.size() + cmdlist._presents.size());
    for (auto &&present : cmdlist._presents) { _presents.emplace_back(present); }
    // clear the other command list
    cmdlist.clear();
    return *this;
}

CommandList &CommandList::operator<<(luisa::unique_ptr<Command> &&cmd) noexcept {
    return append(std::move(cmd));
}

CommandList::CallbackContainer CommandList::steal_callbacks() noexcept {
    return std::move(_callbacks);
}

CommandList::CommandContainer CommandList::steal_commands() noexcept {
    return std::move(_commands);
}

CommandList::PresentContainer CommandList::steal_presents() noexcept {
        return std::move(_presents);
    }
    CommandList::DtorCallbacks CommandList::steal_dtor_callbacks() noexcept {
        return std::move(_dtor_callbacks);
    }
CommandList &CommandList::add_dtor_callback(luisa::move_only_function<void()> &&callback) noexcept {
    _dtor_callbacks.emplace_back(std::move(callback));
    return *this;
}

CommandList &CommandList::add_present(SwapchainPresent &&present) noexcept {
    _presents.emplace_back(present);
    return *this;
}

CommandList CommandList::create(size_t reserved_command_size, size_t reserved_callback_size) noexcept {
    CommandList list{};
    list.reserve(reserved_command_size, reserved_callback_size);
    return list;
}
CommandList::CommandList() noexcept = default;

CommandList::Commit CommandList::commit() noexcept {
    _committed = true;
    return Commit{std::move(*this)};
}
bool CommandList::empty() const noexcept { return _commands.empty() && _callbacks.empty() && _presents.empty(); }
luisa::span<const SwapchainPresent> CommandList::presents() const noexcept { return luisa::span{_presents}; }

CommandList::CommandList(CommandList &&another) noexcept
    : _commands{std::move(another._commands)},
      _callbacks{std::move(another._callbacks)},
      _presents{std::move(another._presents)},
      _dtor_callbacks(std::move(another._dtor_callbacks)),
      _committed{another._committed} { another._committed = false; }

CommandList::CommandList(
    CommandContainer &&commands,
    CallbackContainer &&callbacks,
    PresentContainer &&presents,
    DtorCallbacks &&dtor_callbacks) noexcept
    : _commands{std::move(commands)},
      _callbacks{std::move(callbacks)},
      _presents{std::move(presents)},
      _dtor_callbacks{std::move(dtor_callbacks)} {}
}// namespace luisa::compute
