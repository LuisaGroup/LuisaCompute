#include <luisa/core/logging.h>
#include <luisa/xir/instructions/store.h>

namespace luisa::compute::xir {

StoreInst::StoreInst(Pool *pool, Value *variable,
                     Value *value, const Name *name) noexcept
    : Instruction{pool, nullptr, name} {
    auto oprands = std::array{variable, value};
    LUISA_DEBUG_ASSERT(oprands[operand_index_variable] == variable, "Unexpected operand index.");
    LUISA_DEBUG_ASSERT(oprands[operand_index_value] == value, "Unexpected operand index.");
    set_operands(oprands);
}

void StoreInst::set_variable(Value *variable) noexcept {
    set_operand(operand_index_variable, variable);
}

void StoreInst::set_value(Value *value) noexcept {
    set_operand(operand_index_value, value);
}

}// namespace luisa::compute::xir
