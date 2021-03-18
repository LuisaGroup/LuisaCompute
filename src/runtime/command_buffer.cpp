//
// Created by Mike Smith on 2021/3/18.
//

#include <runtime/command_buffer.h>

namespace luisa::compute {

void CommandBuffer::append(std::unique_ptr<Command> cmd) noexcept {
    _commands.emplace_back(std::move(cmd));
}

void CommandBuffer::set_callback(std::function<void()> cb) noexcept {
    if (_callback) { LUISA_ERROR_WITH_LOCATION("Command buffer callback already set."); }
    _callback = std::move(cb);
}

}
