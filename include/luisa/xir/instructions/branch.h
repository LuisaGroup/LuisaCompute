#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class BasicBlock;

class BranchInst : public Instruction {

private:
    BasicBlock *_true_block = nullptr;
    BasicBlock *_false_block = nullptr;

public:

};

}// namespace luisa::compute::xir
