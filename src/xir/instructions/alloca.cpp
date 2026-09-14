#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/alloca.h>

namespace luisa::compute::xir {

AllocaInst::AllocaInst(BasicBlock *parent_block, const Type *type, AllocaOp op) noexcept
    : Super{op, parent_block, type} {}

AllocaInst *AllocaInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    auto *result = b.alloca_(type(), op());
    result->set_coro_return_selector(_coro_return_selector);
    return result;
}

}// namespace luisa::compute::xir
