#include <luisa/xir/instructions/break.h>

namespace luisa::compute::xir {

BreakInst::BreakInst(Pool *pool, const Name *name) noexcept
    : Instruction{pool, nullptr, name} {}

}// namespace luisa::compute::xir
