//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#include <functional>
#include <runtime/command.h>

namespace luisa::compute {

class CommandGroup {

private:
    std::vector<std::unique_ptr<Command>> _commands;
    std::function<void()> _callback;

public:

};

}
