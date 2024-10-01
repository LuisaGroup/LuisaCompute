#include <luisa/xir/basic_block.h>

namespace luisa::compute::xir {

BasicBlock::BasicBlock(Function *function, BasicBlock *parent_block, const Name *name) noexcept
    : _function{function}, _parent_block{parent_block}, _name{name} {
    _instructions.head_sentinel()->set_parent_block(this);
    _instructions.tail_sentinel()->set_parent_block(this);
}

void BasicBlock::set_function(Function *function) noexcept {
    _function = function;
}

void BasicBlock::set_parent_block(BasicBlock *parent_block) noexcept {
    _parent_block = parent_block;
}

void BasicBlock::set_name(const Name *name) noexcept {
    _name = name;
}

}// namespace luisa::compute::xir
