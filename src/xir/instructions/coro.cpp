#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/coro.h>

namespace luisa::compute::xir {

CoroSuspendInst::CoroSuspendInst(BasicBlock *parent_block, uint32_t token, luisa::string name, Value *frame) noexcept
    : Super{parent_block}, _token{token}, _name{std::move(name)} {
    set_operands(std::array{frame});
}

CoroSuspendInst *CoroSuspendInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    return b.coro_suspend(token(), luisa::string{name()}, resolver.resolve(frame()));
}

CoroResumeInst::CoroResumeInst(BasicBlock *parent_block, uint32_t token, Value *frame) noexcept
    : Super{parent_block, nullptr}, _token{token} {
    set_operands(std::array{frame});
}

CoroResumeInst *CoroResumeInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    return b.coro_resume(token(), resolver.resolve(frame()));
}

CoroTerminateInst::CoroTerminateInst(BasicBlock *parent_block) noexcept
    : Super{parent_block} {}

CoroTerminateInst *CoroTerminateInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    return b.coro_terminate();
}

CoroRegisterInst::CoroRegisterInst(BasicBlock *parent_block, luisa::string name, Value *value, Value *frame) noexcept
    : Super{parent_block, nullptr}, _name{std::move(name)} {
    set_operands(std::array{value, frame});
}

CoroRegisterInst *CoroRegisterInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    return b.coro_register(luisa::string{name()}, resolver.resolve(value()), resolver.resolve(frame()));
}

}
