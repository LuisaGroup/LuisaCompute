#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/coro/token.h>

namespace luisa::compute::xir {

CoroTokenInst::CoroTokenInst(BasicBlock *parent_block) noexcept
    : Super{parent_block, Type::of<uint>()} {}

CoroTokenInst *CoroTokenInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    static_cast<void>(resolver);
    return b.coro_token();
}

}// namespace luisa::compute::xir
