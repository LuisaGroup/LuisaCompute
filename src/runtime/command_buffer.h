//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#include <functional>
#include <runtime/command.h>

namespace luisa::compute {

class CommandBuffer {

private:
    std::vector<std::unique_ptr<Command>> _commands;

public:
    CommandBuffer() noexcept { LUISA_VERBOSE_WITH_LOCATION("Created command buffer."); }
    void append(std::unique_ptr<Command> cmd) noexcept;
    [[nodiscard]] auto begin() const noexcept { return _commands.cbegin(); }
    [[nodiscard]] auto end() const noexcept { return _commands.cend(); }
};

}
