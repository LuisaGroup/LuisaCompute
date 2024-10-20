#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class LC_XIR_API BreakInst final : public Instruction {

public:
    explicit BreakInst(Pool *pool, const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::BREAK;
    }
};

}// namespace luisa::compute::xir
