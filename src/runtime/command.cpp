//
// Created by Mike Smith on 2021/3/3.
//

#include <runtime/command.h>

namespace luisa::compute {

std::span<const KernelLaunchCommand::Argument> KernelLaunchCommand::arguments() const noexcept {
    return _arguments;
}

}
