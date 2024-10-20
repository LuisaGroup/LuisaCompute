//
// Created by Mike on 2024/10/20.
//

#include <luisa/xir/instructions/print.h>

namespace luisa::compute::xir {

PrintInst::PrintInst(Pool *pool, luisa::string format,
                     luisa::span<Value *const> operands,
                     const Name *name) noexcept
    : Instruction{pool, nullptr, name},
      _format{std::move(format)} { set_operands(operands); }

}// namespace luisa::compute::xir
