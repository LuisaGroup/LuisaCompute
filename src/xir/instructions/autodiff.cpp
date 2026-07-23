#include <luisa/xir/function.h>
#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/autodiff.h>

namespace luisa::compute::xir {

AutodiffScopeInst::AutodiffScopeInst(BasicBlock *parent_block, bool forward, size_t n_forward_grads) noexcept
    : Super{parent_block}, _forward{forward}, _n_forward_grads{forward ? n_forward_grads : 0u} {
    LUISA_ASSERT(!forward || n_forward_grads > 0u, "Forward autodiff scope requires at least one gradient slot.");
    set_operands(std::array{static_cast<Value *>(nullptr)});
}

void AutodiffScopeInst::set_forward(bool forward, size_t n_forward_grads) noexcept {
    LUISA_ASSERT(!forward || n_forward_grads > 0u, "Forward autodiff scope requires at least one gradient slot.");
    _forward = forward;
    _n_forward_grads = forward ? n_forward_grads : 0u;
}

void AutodiffScopeInst::set_entry_block(BasicBlock *block) noexcept {
    set_operand(operand_index_entry_block, block);
}

BasicBlock *AutodiffScopeInst::create_entry_block(bool overwrite_existing) noexcept {
    LUISA_ASSERT(entry_block() == nullptr || overwrite_existing, "Entry block already exists.");
    auto new_block = parent_function()->create_basic_block();
    set_entry_block(new_block);
    return new_block;
}

BasicBlock *AutodiffScopeInst::entry_block() noexcept {
    auto block = operand(operand_index_entry_block);
    LUISA_DEBUG_ASSERT(block == nullptr || block->isa<BasicBlock>(), "Invalid autodiff entry block.");
    return static_cast<BasicBlock *>(block);
}

const BasicBlock *AutodiffScopeInst::entry_block() const noexcept {
    return const_cast<AutodiffScopeInst *>(this)->entry_block();
}

AutodiffScopeInst *AutodiffScopeInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    auto cloned = b.autodiff_scope(is_forward(), n_forward_grads());
    auto resolved_entry_block = resolver.resolve(entry_block());
    LUISA_DEBUG_ASSERT(resolved_entry_block == nullptr || resolved_entry_block->isa<BasicBlock>(), "Invalid entry block.");
    cloned->set_entry_block(static_cast<BasicBlock *>(resolved_entry_block));
    auto resolved_merge_block = resolver.resolve(merge_block());
    LUISA_DEBUG_ASSERT(resolved_merge_block == nullptr || resolved_merge_block->isa<BasicBlock>(), "Invalid merge block.");
    cloned->set_merge_block(static_cast<BasicBlock *>(resolved_merge_block));
    return cloned;
}

AutodiffIntrinsicInst::AutodiffIntrinsicInst(BasicBlock *parent_block, const Type *type, AutodiffIntrinsicOp op,
                                             luisa::span<Value *const> operands) noexcept
    : Super{op, parent_block, type} { set_operands(operands); }

AutodiffIntrinsicInst *AutodiffIntrinsicInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 16u> resolved_operands;
    resolved_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        resolved_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return b.call(type(), op(), resolved_operands);
}

}// namespace luisa::compute::xir
