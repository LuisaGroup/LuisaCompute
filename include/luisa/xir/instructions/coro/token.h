#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LUISA_XIR_API CoroTokenInst final : public DerivedInstruction<CoroTokenInst, DerivedInstructionTag::CORO_TOKEN> {
public:
    explicit CoroTokenInst(BasicBlock *parent_block) noexcept;
    [[nodiscard]] CoroTokenInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
