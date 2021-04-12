//
// Created by Mike Smith on 2021/3/18.
//

#include <runtime/command_buffer.h>

namespace luisa::compute {

void CommandBuffer::append(CommandHandle cmd) noexcept {
    _commands.emplace_back(std::move(cmd));
    LUISA_VERBOSE_WITH_LOCATION("Added command to command buffer.");
}

}// namespace luisa::compute
