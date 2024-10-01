#include <luisa/xir/instructions/branch.h>

namespace luisa::compute::xir {

BranchInst::BranchInst(Pool *pool, Value *cond, const Name *name) noexcept
    : Instruction{pool, nullptr, name} {
    auto true_block = static_cast<Value *>(pool->create<BasicBlock>());
    auto false_block = static_cast<Value *>(pool->create<BasicBlock>());
    set_operands(std::array{cond, true_block, false_block});
}

void BranchInst::set_cond(Value *cond) noexcept {
    set_operand(0, cond);
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
