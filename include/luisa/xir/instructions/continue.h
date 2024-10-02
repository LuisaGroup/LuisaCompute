#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class LC_XIR_API ContinueInst : public Instruction {
public:
    explicit ContinueInst(Pool *pool, const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept final {
        return DerivedInstructionTag::CONTINUE;
    }
};

}// namespace luisa::compute::xir