//
// Created by Mike on 2024/10/20.
//

#include <luisa/xir/instructions/intrinsic.h>

namespace luisa::compute::xir {

IntrinsicInst::IntrinsicInst(Pool *pool, IntrinsicOp op,
                             luisa::span<Value *const> operands,
                             const Type *type, const Name *name) noexcept
    : Instruction{pool, type, name}, _op{op} { set_operands(operands); }

}// namespace luisa::compute::xir
