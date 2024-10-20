#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class BasicBlock;

// A loop instruction:
// loop {
//   { prepare_block }
//   if (cond) {
//     { body_block }
//     // continue goes here
//     { update_block }
//   } else {
//     // goto merge
//   }
// }
// // break goes here
// { merge_block }
//
// Note: this instruction must be the terminator of a basic block.
class LC_XIR_API LoopInst final : public Instruction {

public:
    static constexpr size_t operand_index_prepare_block = 0u;
    static constexpr size_t operand_index_cond = 1u;
    static constexpr size_t operand_index_body_block = 2u;
    static constexpr size_t operand_index_update_block = 3u;
    static constexpr size_t operand_index_merge_block = 4u;

public:
    explicit LoopInst(Pool *pool, Value *cond = nullptr,
                      const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::LOOP;
    }

    void set_cond(Value *cond) noexcept;

    [[nodiscard]] BasicBlock *prepare_block() noexcept;
    [[nodiscard]] const BasicBlock *prepare_block() const noexcept;

    [[nodiscard]] Value *cond() noexcept;
    [[nodiscard]] const Value *cond() const noexcept;

    [[nodiscard]] BasicBlock *body_block() noexcept;
    [[nodiscard]] const BasicBlock *body_block() const noexcept;

    [[nodiscard]] BasicBlock *update_block() noexcept;
    [[nodiscard]] const BasicBlock *update_block() const noexcept;

    [[nodiscard]] BasicBlock *merge_block() noexcept;
    [[nodiscard]] const BasicBlock *merge_block() const noexcept;
};

}// namespace luisa::compute::xir
