#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LUISA_XIR_API SuspendInst final : public DerivedInstruction<SuspendInst, DerivedInstructionTag::SUSPEND> {
public:
    uint32_t coro_token;

    explicit SuspendInst(BasicBlock *parent_block, uint32_t token) noexcept;

    [[nodiscard]] SuspendInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
