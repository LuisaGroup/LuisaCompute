#include <luisa/core/logging.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/branch.h>

namespace luisa::compute::xir {

BranchInst::BranchInst(Pool *pool, Value *cond, const Name *name) noexcept
    : Instruction{pool, nullptr, name} {
    auto true_block = static_cast<Value *>(pool->create<BasicBlock>(this));
    auto false_block = static_cast<Value *>(pool->create<BasicBlock>(this));
    auto merge_block = static_cast<Value *>(pool->create<BasicBlock>(this));
    auto operands = std::array{cond, true_block, false_block, merge_block};
    LUISA_DEBUG_ASSERT(operands[operand_index_cond] == cond, "Unexpected operand index.");
    LUISA_DEBUG_ASSERT(operands[operand_index_true_block] == true_block, "Unexpected operand index.");
    LUISA_DEBUG_ASSERT(operands[operand_index_false_block] == false_block, "Unexpected operand index.");
    LUISA_DEBUG_ASSERT(operands[operand_index_merge_block] == merge_block, "Unexpected operand index.");
    set_operands(operands);
}

void BranchInst::set_cond(Value *cond) noexcept {
    LUISA_DEBUG_ASSERT(cond == nullptr || cond->type() == Type::of<bool>(),
                       "Branch condition must be a boolean value.");
    set_operand(operand_index_cond, cond);
}

Value *BranchInst::cond() noexcept {
    return operand(operand_index_cond);
}

const Value *BranchInst::cond() const noexcept {
    return operand(operand_index_cond);
}

BasicBlock *BranchInst::true_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_true_block));
}

const BasicBlock *BranchInst::true_block() const noexcept {
    return static_cast<const BasicBlock *>(operand(operand_index_true_block));
}

BasicBlock *BranchInst::false_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_false_block));
}

const BasicBlock *BranchInst::false_block() const noexcept {
    return static_cast<const BasicBlock *>(operand(operand_index_false_block));
}

BasicBlock *BranchInst::merge_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_merge_block));
}

const BasicBlock *BranchInst::merge_block() const noexcept {
    return static_cast<const BasicBlock *>(operand(operand_index_merge_block));
}

}// namespace luisa::compute::xir
