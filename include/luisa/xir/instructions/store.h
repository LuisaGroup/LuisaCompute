#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API StoreInst final : public Instruction {

public:
    static constexpr size_t operand_index_variable = 0u;
    static constexpr size_t operand_index_value = 1u;

public:
    explicit StoreInst(Pool *pool,
                       Value *variable = nullptr,
                       Value *value = nullptr,
                       const Name *name = nullptr) noexcept;

    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::STORE;
    }

    [[nodiscard]] auto variable() noexcept { return operand(operand_index_variable); }
    [[nodiscard]] auto variable() const noexcept { return operand(operand_index_variable); }
    [[nodiscard]] auto value() noexcept { return operand(operand_index_value); }
    [[nodiscard]] auto value() const noexcept { return operand(operand_index_value); }

    void set_variable(Value *variable) noexcept;
    void set_value(Value *value) noexcept;
};

}// namespace luisa::compute::xir
