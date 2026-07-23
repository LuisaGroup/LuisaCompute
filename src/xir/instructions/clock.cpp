#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/clock.h>

namespace luisa::compute::xir {

ClockInst::ClockInst(BasicBlock *parent_block) noexcept
    : Super{parent_block, Type::of<uint64_t>()} {}

ClockInst *ClockInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    return b.clock();
}

}// namespace luisa::compute::xir
