#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LUISA_XIR_API AutodiffScopeInst final : public ControlFlowMergeMixin<DerivedTerminatorInstruction<AutodiffScopeInst, DerivedInstructionTag::AUTODIFF_SCOPE>> {

public:
    static constexpr size_t operand_index_entry_block = 0u;

private:
    bool _forward{false};
    size_t _n_forward_grads{0u};

public:
    explicit AutodiffScopeInst(BasicBlock *parent_block, bool forward = false, size_t n_forward_grads = 0u) noexcept;
    [[nodiscard]] bool is_forward() const noexcept { return _forward; }
    [[nodiscard]] bool is_reverse() const noexcept { return !_forward; }
    [[nodiscard]] size_t n_forward_grads() const noexcept { return _n_forward_grads; }
    void set_forward(bool forward, size_t n_forward_grads = 0u) noexcept;
    void set_entry_block(BasicBlock *block) noexcept;
    BasicBlock *create_entry_block(bool overwrite_existing = false) noexcept;
    [[nodiscard]] BasicBlock *entry_block() noexcept;
    [[nodiscard]] const BasicBlock *entry_block() const noexcept;
    [[nodiscard]] AutodiffScopeInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

class LUISA_XIR_API AutodiffIntrinsicInst final : public InstructionOpMixin<AutodiffIntrinsicOp, DerivedInstruction<AutodiffIntrinsicInst, DerivedInstructionTag::AUTODIFF_INTRINSIC>> {
public:
    AutodiffIntrinsicInst(BasicBlock *parent_block, const Type *type, AutodiffIntrinsicOp op,
                          luisa::span<Value *const> operands = {}) noexcept;
    [[nodiscard]] AutodiffIntrinsicInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
