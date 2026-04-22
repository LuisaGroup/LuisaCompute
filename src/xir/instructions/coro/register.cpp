#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/coro/register.h>

namespace luisa::compute::xir {

CoroRegisterInst::CoroRegisterInst(BasicBlock *parent_block, Value *value, luisa::string name) noexcept
    : Super{std::move(name), parent_block, nullptr} {
    auto operands = std::array{value};
    set_operands(operands);
}

Value *CoroRegisterInst::value() noexcept {
    return operand(operand_index_value);
}

const Value *CoroRegisterInst::value() const noexcept {
    return const_cast<CoroRegisterInst *>(this)->value();
}

void CoroRegisterInst::set_value(Value *value) noexcept {
    set_operand(operand_index_value, value);
}

CoroRegisterInst *CoroRegisterInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    return b.coro_register(resolver.resolve(value()), luisa::string{name()});
}

}// namespace luisa::compute::xir
