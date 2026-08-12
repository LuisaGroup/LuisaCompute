#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/core/logging.h>

namespace luisa::compute::xir {

CoroSuspendInst::CoroSuspendInst(BasicBlock *parent_block, uint32_t token, luisa::string name, Value *frame) noexcept
    : CoroSuspendInst{parent_block, token, std::move(name), frame,
                      {}, {}} {}

CoroSuspendInst::CoroSuspendInst(
    BasicBlock *parent_block, uint32_t token, luisa::string name,
    Value *frame, luisa::span<const luisa::string> frame_export_names,
    luisa::span<Value *const> frame_export_values) noexcept
    : Super{parent_block}, _token{token}, _name{std::move(name)},
      _frame_export_names{frame_export_names.begin(),
                          frame_export_names.end()} {
    LUISA_ASSERT(
        frame_export_names.size() == frame_export_values.size(),
        "Coroutine suspension frame export name/value counts differ.");
    reserve_operands(operand_index_frame_export_offset +
                     frame_export_values.size());
    add_operand(frame);
    for (auto *value : frame_export_values) { add_operand(value); }
}

CoroSuspendInst *CoroSuspendInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    luisa::vector<Value *> values;
    values.reserve(frame_export_count());
    for (size_t i = 0u; i < frame_export_count(); ++i) {
        values.emplace_back(resolver.resolve(frame_export_value(i)));
    }
    return b.coro_suspend(
        token(), luisa::string{name()}, resolver.resolve(frame()),
        frame_export_names(), luisa::span{values});
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

}
