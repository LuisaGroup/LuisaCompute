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
    std::function<void()> _callback;

public:
    void append(std::unique_ptr<Command> cmd) noexcept;
    void set_callback(std::function<void()> cb) noexcept;
    [[nodiscard]] auto begin() const noexcept { return _commands.cbegin(); }
    [[nodiscard]] auto end() const noexcept { return _commands.cend(); }
    [[nodiscard]] auto &callback() noexcept { return _callback; }
    [[nodiscard]] const auto &callback() const noexcept { return _callback; }
};

}
