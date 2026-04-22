#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/coro/id.h>

namespace luisa::compute::xir {

CoroIdInst::CoroIdInst(BasicBlock *parent_block) noexcept
    : Super{parent_block, Type::of<uint3>()} {}

CoroIdInst *CoroIdInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    static_cast<void>(resolver);
    return b.coro_id();
}

}// namespace luisa::compute::xir
