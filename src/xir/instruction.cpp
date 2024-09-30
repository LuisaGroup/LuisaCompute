#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

Instruction::Instruction(BasicBlock *parent_block) noexcept
    : _parent_block{parent_block} {}

void Instruction::remove_self() noexcept {
    // remove the node from the instruction list
    IntrusiveNode::remove_self();
    // also remove the operand uses
    remove_operand_uses();
}

}// namespace luisa::compute::xir
