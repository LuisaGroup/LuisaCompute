#pragma once

#include <cstdint>

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Switch instruction:
//
// switch (value) {
//   case case_values[0]: { case_blocks[0] }
//   case case_values[1]: { case_blocks[1] }
//   ...
//   default: { default_block }
// }
// { merge_block }
//
// Note: this instruction must be the terminator of a basic block.
class LUISA_XIR_API SwitchInst final : public ControlFlowMergeMixin<DerivedTerminatorInstruction<SwitchInst, DerivedInstructionTag::SWITCH>> {

public:
    using case_value_type = uint64_t;
    static constexpr size_t operand_index_value = 0u;
    static constexpr size_t operand_index_default_block = 1u;
    static constexpr size_t operand_index_case_block_offset = 2u;

private:
    luisa::vector<case_value_type> _case_values;

public:
    SwitchInst(BasicBlock *parent_block, Value *value) noexcept;

    [[nodiscard]] static case_value_type canonicalize_case_value(
        const Type *selector_type, case_value_type value) noexcept;

    void set_value(Value *value) noexcept;
    void set_default_block(BasicBlock *block) noexcept;

    BasicBlock *create_default_block(bool overwrite_existing = false) noexcept;
    BasicBlock *create_case_block(case_value_type value) noexcept;

    void set_case_count(size_t count) noexcept;
    [[nodiscard]] size_t case_count() const noexcept;

    void set_case(size_t index, case_value_type value, BasicBlock *block) noexcept;
    void set_case_value(size_t index, case_value_type value) noexcept;
    void set_case_block(size_t index, BasicBlock *block) noexcept;
    void add_case(case_value_type value, BasicBlock *block) noexcept;
    void insert_case(size_t index, case_value_type value, BasicBlock *block) noexcept;
    void remove_case(size_t index) noexcept;

    [[nodiscard]] case_value_type case_value(size_t index) const noexcept;
    [[nodiscard]] BasicBlock *case_block(size_t index) noexcept;
    [[nodiscard]] const BasicBlock *case_block(size_t index) const noexcept;

    [[nodiscard]] luisa::span<const case_value_type> case_values() const noexcept;
    [[nodiscard]] luisa::span<Use *> case_block_uses() noexcept;
    [[nodiscard]] luisa::span<const Use *const> case_block_uses() const noexcept;

    [[nodiscard]] Value *value() noexcept;
    [[nodiscard]] const Value *value() const noexcept;

    [[nodiscard]] BasicBlock *default_block() noexcept;
    [[nodiscard]] const BasicBlock *default_block() const noexcept;

    [[nodiscard]] SwitchInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
