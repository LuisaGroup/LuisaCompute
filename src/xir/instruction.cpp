#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

Instruction::Instruction(Pool *pool, const Type *type, const Name *name) noexcept
    : Super{pool, type, name} {}

void Instruction::set_parent_block(BasicBlock *block) noexcept {
    _parent_block = block;
}

void Instruction::remove_self() noexcept {
    Super::remove_self();
    set_parent_block(nullptr);
    _remove_operand_uses();
}

void Instruction::insert_before_self(Instruction *node) noexcept {
    Super::insert_before_self(node);
    node->set_parent_block(_parent_block);
    node->_add_operand_uses();
}

void Instruction::insert_after_self(Instruction *node) noexcept {
    Super::insert_after_self(node);
    node->set_parent_block(_parent_block);
    node->_add_operand_uses();
}

}// namespace luisa::compute::xir
