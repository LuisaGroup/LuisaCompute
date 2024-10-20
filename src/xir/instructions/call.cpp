#include <luisa/xir/instructions/call.h>

namespace luisa::compute::xir {

CallInst::CallInst(Pool *pool, Value *callee,
                   luisa::span<Value *const> arguments,
                   const Type *type, const Name *name) noexcept
    : Instruction{pool, type, name} {
    set_operand_count(1u + arguments.size());
    set_operand(operand_index_callee, callee);
    for (auto i = 0u; i < arguments.size(); i++) {
        set_operand(operand_index_argument_offset + i, arguments[i]);
    }
}

void CallInst::set_callee(Value *callee) noexcept {
    set_operand(operand_index_callee, callee);
}

void CallInst::set_arguments(luisa::span<Value *const> arguments) noexcept {
    set_operand_count(1u + arguments.size());
    for (auto i = 0u; i < arguments.size(); i++) {
        set_operand(operand_index_argument_offset + i, arguments[i]);
    }
}

void CallInst::set_argument(size_t index, Value *argument) noexcept {
    set_operand(operand_index_argument_offset + index, argument);
}

void CallInst::add_argument(Value *argument) noexcept {
    add_operand(argument);
}

void CallInst::insert_argument(size_t index, Value *argument) noexcept {
    insert_operand(operand_index_argument_offset + index, argument);
}

void CallInst::remove_argument(size_t index) noexcept {
    remove_operand(operand_index_argument_offset + index);
}

}// namespace luisa::compute::xir
