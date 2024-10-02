#include <luisa/core/logging.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/loop.h>

namespace luisa::compute::xir {

LoopInst::LoopInst(Pool *pool, Value *cond, const Name *name) noexcept
    : Instruction{pool, nullptr, name} {
    auto prepare = static_cast<Value *>(pool->create<BasicBlock>(this));
    auto body = static_cast<Value *>(pool->create<BasicBlock>(this));
    auto update = static_cast<Value *>(pool->create<BasicBlock>(this));
    auto merge = static_cast<Value *>(pool->create<BasicBlock>(this));
    auto operands = std::array{prepare, cond, body, update, merge};
    LUISA_DEBUG_ASSERT(operands[operand_index_cond] == cond, "Unexpected operand index.");
    LUISA_DEBUG_ASSERT(operands[operand_index_prepare_block] == prepare, "Unexpected operand index.");
    LUISA_DEBUG_ASSERT(operands[operand_index_body_block] == body, "Unexpected operand index.");
    LUISA_DEBUG_ASSERT(operands[operand_index_update_block] == update, "Unexpected operand index.");
    LUISA_DEBUG_ASSERT(operands[operand_index_merge_block] == merge, "Unexpected operand index.");
    set_operands(operands);
}

void LoopInst::set_cond(Value *cond) noexcept {
    LUISA_DEBUG_ASSERT(cond == nullptr || cond->type() == Type::of<bool>(),
                       "Loop condition must be a boolean value.");
    set_operand(operand_index_cond, cond);
}

BasicBlock *LoopInst::prepare_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_prepare_block));
}

const BasicBlock *LoopInst::prepare_block() const noexcept {
    return const_cast<LoopInst *>(this)->prepare_block();
}

Value *LoopInst::cond() noexcept {
    return operand(operand_index_cond);
}

const Value *LoopInst::cond() const noexcept {
    return operand(operand_index_cond);
}

BasicBlock *LoopInst::body_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_body_block));
}

const BasicBlock *LoopInst::body_block() const noexcept {
    return const_cast<LoopInst *>(this)->body_block();
}

BasicBlock *LoopInst::update_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_update_block));
}

const BasicBlock *LoopInst::update_block() const noexcept {
    return const_cast<LoopInst *>(this)->update_block();
}

BasicBlock *LoopInst::merge_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_merge_block));
}

const BasicBlock *LoopInst::merge_block() const noexcept {
    return const_cast<LoopInst *>(this)->merge_block();
}

}// namespace luisa::compute::xir
