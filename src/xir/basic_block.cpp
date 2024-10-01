#include <luisa/xir/basic_block.h>

namespace luisa::compute::xir {

BasicBlock::BasicBlock(Pool *pool, const Name *name) noexcept
    : Value{pool, nullptr, name}, _instructions{pool} {
    _instructions.head_sentinel()->_set_parent_block(this);
    _instructions.tail_sentinel()->_set_parent_block(this);
}

Instruction *BasicBlock::parent_instruction() noexcept {
    return use_list().empty() ? nullptr : static_cast<Instruction *>(use_list().front().user());
}

const Instruction *BasicBlock::parent_instruction() const noexcept {
    return const_cast<BasicBlock *>(this)->parent_instruction();
}

BasicBlock *BasicBlock::parent_block() noexcept {
    auto parent_inst = parent_instruction();
    return parent_inst ? parent_inst->parent_block() : nullptr;
}

const BasicBlock *BasicBlock::parent_block() const noexcept {
    return const_cast<BasicBlock *>(this)->parent_block();
}

}// namespace luisa::compute::xir
