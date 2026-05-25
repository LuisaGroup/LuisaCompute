#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/coroutine.h>

namespace luisa::compute::xir {

CoroRegisterInst::CoroRegisterInst(BasicBlock *parent_block, Value *value, luisa::string name) noexcept
    : Super{parent_block, nullptr}, _name{std::move(name)} {
    set_operand_count(1u);
    set_value(value);
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
    return b.coro_register(resolver.resolve(value()), luisa::string{_name});
}

CoroSuspendInst::CoroSuspendInst(BasicBlock *parent_block, uint32_t token) noexcept
    : Super{parent_block, nullptr}, _token{token} {}

CoroSuspendInst *CoroSuspendInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    return b.coro_suspend(_token);
}

}
