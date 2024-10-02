#include <luisa/xir/basic_block.h>

namespace luisa::compute::xir {

BasicBlock::BasicBlock(Pool *pool, Value *parent_value, const Name *name) noexcept
    : Value{pool, nullptr, name},
      _parent_value{parent_value},
      _instructions{pool} {
    _instructions.head_sentinel()->set_parent_block(this);
    _instructions.tail_sentinel()->set_parent_block(this);
}

void BasicBlock::set_parent_value(Value *parent_value) noexcept {
    _parent_value = parent_value;
}

}// namespace luisa::compute::xir
