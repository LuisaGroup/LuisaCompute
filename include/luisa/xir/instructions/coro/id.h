#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LUISA_XIR_API CoroIdInst final : public DerivedInstruction<CoroIdInst, DerivedInstructionTag::CORO_ID> {
public:
    explicit CoroIdInst(BasicBlock *parent_block) noexcept;
    [[nodiscard]] CoroIdInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
