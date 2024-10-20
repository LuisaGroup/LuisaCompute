#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API CallInst final : public Instruction {

public:
    static constexpr size_t operand_index_callee = 0u;
    static constexpr size_t operand_index_argument_offset = 1u;

public:
    explicit CallInst(Pool *pool, Value *callee = nullptr,
                      luisa::span<Value *const> arguments = {},
                      const Type *type = nullptr,
                      const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::CALL;
    }
    [[nodiscard]] auto callee() noexcept { return operand(operand_index_callee); }
    [[nodiscard]] auto callee() const noexcept { return operand(operand_index_callee); }
    [[nodiscard]] auto argument(size_t index) noexcept { return operand(operand_index_argument_offset + index); }
    [[nodiscard]] auto argument(size_t index) const noexcept { return operand(operand_index_argument_offset + index); }
    [[nodiscard]] auto argument_uses() noexcept { return operand_uses().subspan(operand_index_argument_offset); }
    [[nodiscard]] auto argument_uses() const noexcept { return operand_uses().subspan(operand_index_argument_offset); }
    [[nodiscard]] auto argument_count() const noexcept { return argument_uses().size(); }

    void set_callee(Value *callee) noexcept;
    void set_arguments(luisa::span<Value *const> arguments) noexcept;
    void set_argument(size_t index, Value *argument) noexcept;
    void add_argument(Value *argument) noexcept;
    void insert_argument(size_t index, Value *argument) noexcept;
    void remove_argument(size_t index) noexcept;
};

}// namespace luisa::compute::xir
