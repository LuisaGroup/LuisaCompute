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
    : CoroSuspendInst{parent_block, token, std::move(name), frame,
                      frame_export_names, frame_export_values,
                      {}, {}} {}

CoroSuspendInst::CoroSuspendInst(
    BasicBlock *parent_block, uint32_t token, luisa::string name,
    Value *frame, luisa::span<const luisa::string> frame_export_names,
    luisa::span<Value *const> frame_export_values,
    luisa::vector<CoroSuspendExtensionPtr> extensions,
    luisa::span<Value *const> extension_binding_values) noexcept
    : Super{parent_block}, _token{token}, _name{std::move(name)},
      _frame_export_names{frame_export_names.begin(),
                          frame_export_names.end()},
      _extensions{std::move(extensions)},
      _extension_binding_value_count{
          extension_binding_values.size()} {
    LUISA_ASSERT(
        frame_export_names.size() == frame_export_values.size(),
        "Coroutine suspension frame export name/value counts differ.");
    reserve_operands(operand_index_frame_export_offset +
                     frame_export_values.size() +
                     extension_binding_values.size());
    add_operand(frame);
    for (auto *value : frame_export_values) { add_operand(value); }
    for (auto *value : extension_binding_values) { add_operand(value); }
}

CoroSuspendInst *CoroSuspendInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    luisa::vector<Value *> values;
    values.reserve(frame_export_count());
    for (size_t i = 0u; i < frame_export_count(); ++i) {
        values.emplace_back(resolver.resolve(frame_export_value(i)));
    }
    luisa::vector<CoroSuspendExtensionPtr> extensions;
    extensions.reserve(_extensions.size());
    for (auto &&extension : _extensions) {
        extensions.emplace_back(extension->clone());
    }
    luisa::vector<Value *> binding_values;
    binding_values.reserve(extension_binding_value_count());
    for (size_t i = 0u; i < extension_binding_value_count(); ++i) {
        binding_values.emplace_back(
            resolver.resolve(extension_binding_value(i)));
    }
    return b.coro_suspend(
        token(), luisa::string{name()}, resolver.resolve(frame()),
        frame_export_names(), luisa::span{values},
        std::move(extensions), luisa::span{binding_values});
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
