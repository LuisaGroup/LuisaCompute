#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

Instruction::Instruction(const Type *type, BasicBlock *parent_block, const Name *name) noexcept
    : Super{type, name} { set_parent_block(parent_block); }

void Instruction::remove_self() noexcept {
    Super::remove_self();
    remove_operand_uses();
}

void Instruction::insert_before_self(Instruction *node) noexcept {
    Super::insert_before_self(node);
    node->add_operand_uses();
}

void Instruction::insert_after_self(Instruction *node) noexcept {
    Super::insert_after_self(node);
    node->add_operand_uses();
}

}// namespace luisa::compute::xir
