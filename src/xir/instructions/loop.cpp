#include <luisa/xir/basic_block.h>
#include <luisa/xir/instructions/loop.h>

namespace luisa::compute::xir {

LoopInst::LoopInst(Pool *pool, Value *cond, const Name *name) noexcept
    : Instruction{pool, nullptr, name} {
    auto body = static_cast<Value *>(pool->create<BasicBlock>());
    set_operands(std::array{body, cond});
}

void LoopInst::set_cond(Value *cond) noexcept {
    set_operand(1u, cond);
}

Value *LoopInst::cond() noexcept {
    return operand(1u);
}

const Value *LoopInst::cond() const noexcept {
    return operand(1u);
}

BasicBlock *LoopInst::body() noexcept {
    return static_cast<BasicBlock *>(operand(0u));
}

const BasicBlock *LoopInst::body() const noexcept {
    return const_cast<LoopInst *>(this)->body();
}

}// namespace luisa::compute::xir