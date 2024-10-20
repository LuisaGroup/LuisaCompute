#pragma once

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
class LC_XIR_API SwitchInst final : public Instruction {

public:
    using case_value_type = int64_t;
    static constexpr size_t operand_index_value = 0u;
    static constexpr size_t operand_index_merge_block = 1u;
    static constexpr size_t operand_index_default_block = 2u;
    static constexpr size_t operand_index_case_block_offset = 3u;

private:
    luisa::vector<case_value_type> _case_values;

public:
    explicit SwitchInst(Pool *pool, Value *value = nullptr,
                        const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::SWITCH;
    }

    void set_value(Value *value) noexcept;

    void set_case_count(size_t count) noexcept;
    [[nodiscard]] size_t case_count() const noexcept;

    void set_case_value(size_t index, case_value_type value) noexcept;
    BasicBlock *add_case(case_value_type value) noexcept;
    BasicBlock *insert_case(size_t index, case_value_type value) noexcept;
    void remove_case(size_t index) noexcept;

    [[nodiscard]] case_value_type case_value(size_t index) const noexcept;
    [[nodiscard]] BasicBlock *case_block(size_t index) noexcept;
    [[nodiscard]] const BasicBlock *case_block(size_t index) const noexcept;

    [[nodiscard]] luisa::span<const case_value_type> case_values() const noexcept;
    [[nodiscard]] luisa::span<Use *> case_block_uses() noexcept;
    [[nodiscard]] luisa::span<const Use *const> case_block_uses() const noexcept;

    [[nodiscard]] Value *value() noexcept;
    [[nodiscard]] const Value *value() const noexcept;
    [[nodiscard]] BasicBlock *merge_block() noexcept;
    [[nodiscard]] const BasicBlock *merge_block() const noexcept;
    [[nodiscard]] BasicBlock *default_block() noexcept;
    [[nodiscard]] const BasicBlock *default_block() const noexcept;
};

}// namespace luisa::compute::xir
