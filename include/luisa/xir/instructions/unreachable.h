#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class LC_XIR_API UnreachableInst final : public Instruction {
public:
    using Instruction::Instruction;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::UNREACHABLE;
    }
};

}// namespace luisa::compute::xir
