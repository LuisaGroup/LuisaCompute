#include <luisa/xir/basic_block.h>

namespace luisa::compute::xir {

BasicBlock::BasicBlock(Pool *pool, Function *function, const Name *name) noexcept
    : Value{pool, nullptr, name},
      _function{function},
      _instructions{pool} {
    _instructions.head_sentinel()->set_parent_block(this);
    _instructions.tail_sentinel()->set_parent_block(this);
}

void BasicBlock::set_function(Function *function) noexcept {
    _function = function;
}

}// namespace luisa::compute::xir
