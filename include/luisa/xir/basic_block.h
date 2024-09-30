#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class Function;
class Instruction;

class LC_XIR_API BasicBlock : public Value {

private:
    Function *_function = nullptr;
    InlineInstructionList _instructions;

public:
};

}// namespace luisa::compute::xir
