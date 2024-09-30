#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

Instruction::Instruction(BasicBlock *parent_block) noexcept
    : _parent_block{parent_block} {}

void Instruction::remove_self() noexcept {
    IntrusiveNode::remove_self();
    remove_operand_uses();
}

void Instruction::insert_before_self(Instruction *node) noexcept {
    IntrusiveNode::insert_before_self(node);
    node->add_operand_uses();
}

void Instruction::insert_after_self(Instruction *node) noexcept {
    IntrusiveNode::insert_after_self(node);
    node->add_operand_uses();
}

}// namespace luisa::compute::xir
