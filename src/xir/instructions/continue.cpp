#include <luisa/xir/instructions/continue.h>

namespace luisa::compute::xir {

ContinueInst::ContinueInst(Pool *pool, const Name *name) noexcept
    : Instruction{pool, nullptr, name} {}

}// namespace luisa::compute::xir
