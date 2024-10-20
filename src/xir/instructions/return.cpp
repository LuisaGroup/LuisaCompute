#include <luisa/xir/instructions/return.h>

namespace luisa::compute::xir {

ReturnInst::ReturnInst(Pool *pool, Value *value, const Name *name) noexcept
    : Instruction{pool, nullptr, name} {
    set_operands(std::array{value});
}

void ReturnInst::set_return_value(Value *value) noexcept {
    set_operand(operand_index_return_value, value);
}

Value *ReturnInst::return_value() noexcept {
    return operand(operand_index_return_value);
}

const Value *ReturnInst::return_value() const noexcept {
    return operand(operand_index_return_value);
}

const Type *ReturnInst::return_type() const noexcept {
    auto ret_value = return_value();
    return ret_value == nullptr ? nullptr : ret_value->type();
}

}// namespace luisa::compute::xir
