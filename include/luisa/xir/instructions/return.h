#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class LC_XIR_API ReturnInst final : public Instruction {

public:
    static constexpr size_t operand_index_return_value = 0u;

public:
    explicit ReturnInst(Pool *pool, Value *value = nullptr,
                        const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::RETURN;
    }

    // nullptr for void return
    void set_return_value(Value *value) noexcept;
    [[nodiscard]] Value *return_value() noexcept;
    [[nodiscard]] const Value *return_value() const noexcept;
    [[nodiscard]] const Type *return_type() const noexcept;
};

}// namespace luisa::compute::xir
