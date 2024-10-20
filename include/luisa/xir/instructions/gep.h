#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Get element pointer instruction.
class LC_XIR_API GEPInst final : public Instruction {

public:
    static constexpr size_t operand_index_base = 0u;
    static constexpr size_t operand_index_index_offset = 1u;

public:
    explicit GEPInst(Pool *pool, Value *base = nullptr,
                     luisa::span<Value *const> indices = {},
                     const Type *type = nullptr,
                     const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::GEP;
    }
    [[nodiscard]] auto base() noexcept { return operand(operand_index_base); }
    [[nodiscard]] auto base() const noexcept { return operand(operand_index_base); }
    [[nodiscard]] auto index(size_t i) noexcept { return operand(operand_index_index_offset + i); }
    [[nodiscard]] auto index(size_t i) const noexcept { return operand(operand_index_index_offset + i); }
    [[nodiscard]] auto index_uses() noexcept { return operand_uses().subspan(operand_index_index_offset); }
    [[nodiscard]] auto index_uses() const noexcept { return operand_uses().subspan(operand_index_index_offset); }
    [[nodiscard]] auto index_count() const noexcept { return index_uses().size(); }

    void set_base(Value *base) noexcept;
    void set_indices(luisa::span<Value *const> indices) noexcept;
    void set_index(size_t i, Value *index) noexcept;
    void add_index(Value *index) noexcept;
    void insert_index(size_t i, Value *index) noexcept;
    void remove_index(size_t i) noexcept;
};

}// namespace luisa::compute::xir
