#include <luisa/xir/instructions/branch.h>

namespace luisa::compute::xir {

BranchInst::BranchInst(Pool *pool, Value *cond,
                       BasicBlock *true_block,
                       BasicBlock *false_block,
                       BasicBlock *parent_block,
                       const Name *name) noexcept
    : Instruction{pool, nullptr, parent_block, name} {
    set_operand_count(3u);
    set_cond(cond);
    set_true_block(true_block);
    set_false_block(false_block);
}

void BranchInst::set_cond(Value *cond) noexcept {
    set_operand(0, cond);
}

void BranchInst::set_true_block(BasicBlock *block) noexcept {
    set_operand(1, block);
}

void BranchInst::set_false_block(BasicBlock *block) noexcept {
    set_operand(2, block);
}

Value *BranchInst::cond() noexcept {
    return operand(0);
}

const Value *BranchInst::cond() const noexcept {
    return operand(0);
}

BasicBlock *BranchInst::true_block() noexcept {
    return static_cast<BasicBlock *>(operand(1));
}

const BasicBlock *BranchInst::true_block() const noexcept {
    return static_cast<const BasicBlock *>(operand(1));
}

BasicBlock *BranchInst::false_block() noexcept {
    return static_cast<BasicBlock *>(operand(2));
}

const BasicBlock *BranchInst::false_block() const noexcept {
    return static_cast<const BasicBlock *>(operand(2));
}

}// namespace luisa::compute::xir
