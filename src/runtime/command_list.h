//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#include <nlohmann/json.hpp>

#include <core/stl.h>
#include <runtime/command.h>
#include <api/common.h>
namespace luisa::compute {
namespace detail {
    class CommandListConverter;
}
class LC_RUNTIME_API CommandList : concepts::Noncopyable {

private:
    luisa::vector<Command *> _commands;

    // For backends that use C API only
    // DO NOT USE THIS FIELD OTHERWISE
    luisa::optional<LCCommandList> _c_list;
    friend class detail::CommandListConverter;
private:
    void _recycle() noexcept;

public:
    CommandList() noexcept = default;
    ~CommandList() noexcept;
    CommandList(CommandList &&) noexcept;
    CommandList &operator=(CommandList &&rhs) noexcept;
    void reserve(size_t size) noexcept;
    void append(Command *cmd) noexcept;
    void clear() noexcept { _commands.clear(); }
    [[nodiscard]] luisa::vector<Command *> steal_commands() noexcept;
    [[nodiscard]] auto begin() const noexcept { return _commands.begin(); }
    [[nodiscard]] auto end() const noexcept { return _commands.end(); }
    [[nodiscard]] auto empty() const noexcept { return _commands.empty(); }
    [[nodiscard]] auto size() const noexcept { return _commands.size(); }
    [[nodiscard]] auto as_ptr() const noexcept {return _commands.data();}
    // for debug
    [[nodiscard]] nlohmann::json dump_json() const noexcept;
};

}// namespace luisa::compute
