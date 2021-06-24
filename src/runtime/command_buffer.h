//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#include <functional>
#include <runtime/command.h>

namespace luisa::compute {

class CommandBuffer {

private:
    std::vector<CommandHandle> _commands;

public:
    CommandBuffer() noexcept = default;
    void append(CommandHandle cmd) noexcept;
    [[nodiscard]] auto begin() const noexcept { return _commands.cbegin(); }
    [[nodiscard]] auto end() const noexcept { return _commands.cend(); }
    [[nodiscard]] auto empty() const noexcept { return _commands.empty(); }
    [[nodiscard]] auto size() const noexcept { return _commands.size(); }
};

}// namespace luisa::compute
