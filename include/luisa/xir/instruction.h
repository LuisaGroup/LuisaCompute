#pragma once

#include <luisa/xir/user.h>

namespace luisa::compute::xir {

class BasicBlock;

class LC_XIR_API Instruction : public IntrusiveNode<Instruction, User> {

private:
    BasicBlock *_block = nullptr;
};

using InstructionList = IntrusiveList<Instruction>;

}// namespace luisa::compute::xir
