#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LUISA_XIR_API AllocaInst final : public InstructionOpMixin<AllocaOp, DerivedInstruction<AllocaInst, DerivedInstructionTag::ALLOCA>> {
private:
    // Nonzero only for the static activation's return selector. Keeping this
    // slot explicit preserves matched call/return analysis through promotion.
    uint32_t _coro_return_selector{0u};
public:
    [[nodiscard]] uint32_t coro_return_selector() const noexcept { return _coro_return_selector; }
    void set_coro_return_selector(uint32_t id) noexcept { _coro_return_selector = id; }
    AllocaInst(BasicBlock *parent_block, const Type *type, AllocaOp op) noexcept;
    [[nodiscard]] auto is_local() const noexcept { return op() == AllocaOp::LOCAL; }
    [[nodiscard]] auto is_shared() const noexcept { return op() == AllocaOp::SHARED; }
    [[nodiscard]] bool is_lvalue() const noexcept override { return true; }
    [[nodiscard]] AllocaInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
