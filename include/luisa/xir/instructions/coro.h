#pragma once

#include <luisa/ast/coro_suspend.h>
#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class LUISA_XIR_API CoroSuspendInst final : public DerivedTerminatorInstruction<CoroSuspendInst, DerivedInstructionTag::CORO_SUSPEND> {

private:
    uint32_t _token;
    luisa::string _name;
    // Scheduler-visible values are part of the coroutine ABI rather than
    // diagnostic names attached to ordinary SSA values. Operand zero remains
    // the optional materialized frame; operands [1, n) are explicitly named
    // values. Distillation makes each exported value resident on this exact
    // suspension edge, where a scheduler can inspect the waiting frame before
    // the target continuation resumes.
    luisa::vector<luisa::string> _frame_export_names;
    // The complete normalized extension objects are retained here. Their
    // CoroSuspendBinding indices resolve into the trailing owner operands;
    // bindings are only one part of each extension's schema.
    luisa::vector<CoroSuspendExtensionPtr> _extensions;
    size_t _extension_binding_value_count{0u};

public:
    static constexpr size_t operand_index_frame = 0u;
    static constexpr size_t operand_index_frame_export_offset = 1u;

public:
    CoroSuspendInst(BasicBlock *parent_block, uint32_t token, luisa::string name, Value *frame) noexcept;
    CoroSuspendInst(BasicBlock *parent_block, uint32_t token,
                    luisa::string name, Value *frame,
                    luisa::span<const luisa::string> frame_export_names,
                    luisa::span<Value *const> frame_export_values) noexcept;
    CoroSuspendInst(
        BasicBlock *parent_block, uint32_t token,
        luisa::string name, Value *frame,
        luisa::span<const luisa::string> frame_export_names,
        luisa::span<Value *const> frame_export_values,
        luisa::vector<CoroSuspendExtensionPtr> extensions,
        luisa::span<Value *const> extension_binding_values) noexcept;

    [[nodiscard]] auto token() const noexcept { return _token; }
    [[nodiscard]] const luisa::string &name() const noexcept { return _name; }
    [[nodiscard]] auto frame() noexcept { return operand(operand_index_frame); }
    [[nodiscard]] auto frame() const noexcept { return operand(operand_index_frame); }
    [[nodiscard]] auto frame_export_count() const noexcept {
        return _frame_export_names.size();
    }
    [[nodiscard]] auto frame_export_names() const noexcept {
        return luisa::span<const luisa::string>{_frame_export_names};
    }
    [[nodiscard]] auto frame_export_name(size_t index) const noexcept
        -> const luisa::string & {
        return _frame_export_names[index];
    }
    [[nodiscard]] auto frame_export_value(size_t index) noexcept {
        return operand(operand_index_frame_export_offset + index);
    }
    [[nodiscard]] auto frame_export_value(size_t index) const noexcept {
        return operand(operand_index_frame_export_offset + index);
    }
    [[nodiscard]] auto extensions() const noexcept {
        return luisa::span<const CoroSuspendExtensionPtr>{_extensions};
    }
    [[nodiscard]] auto extension_binding_value_count() const noexcept {
        return _extension_binding_value_count;
    }
    [[nodiscard]] auto operand_index_extension_binding_offset() const noexcept {
        return operand_index_frame_export_offset + frame_export_count();
    }
    [[nodiscard]] auto extension_binding_value(size_t index) noexcept {
        return operand(operand_index_extension_binding_offset() + index);
    }
    [[nodiscard]] auto extension_binding_value(size_t index) const noexcept {
        return operand(operand_index_extension_binding_offset() + index);
    }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) noexcept {
        return visitor(*this);
    }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) const noexcept {
        return visitor(*this);
    }

    [[nodiscard]] CoroSuspendInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

class LUISA_XIR_API CoroResumeInst final : public DerivedInstruction<CoroResumeInst, DerivedInstructionTag::CORO_RESUME> {

private:
    uint32_t _token;

public:
    static constexpr size_t operand_index_frame = 0u;

public:
    CoroResumeInst(BasicBlock *parent_block, uint32_t token, Value *frame) noexcept;

    [[nodiscard]] auto token() const noexcept { return _token; }
    [[nodiscard]] auto frame() noexcept { return operand(operand_index_frame); }
    [[nodiscard]] auto frame() const noexcept { return operand(operand_index_frame); }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) noexcept {
        return visitor(*this);
    }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) const noexcept {
        return visitor(*this);
    }

    [[nodiscard]] CoroResumeInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

class LUISA_XIR_API CoroTerminateInst final : public DerivedTerminatorInstruction<CoroTerminateInst, DerivedInstructionTag::CORO_TERMINATE> {

public:
    explicit CoroTerminateInst(BasicBlock *parent_block) noexcept;

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) noexcept {
        return visitor(*this);
    }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) const noexcept {
        return visitor(*this);
    }

    [[nodiscard]] CoroTerminateInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
