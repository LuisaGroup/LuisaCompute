#include <luisa/xir/instructions/coro/suspend.h>
#include <luisa/xir/builder.h>

namespace luisa::compute::xir {

SuspendInst::SuspendInst(BasicBlock *parent_block, uint32_t token) noexcept
    : Super{parent_block, nullptr}, coro_token{token} {}

SuspendInst *SuspendInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    return b.suspend_(coro_token);
}

}// namespace luisa::compute::xir
