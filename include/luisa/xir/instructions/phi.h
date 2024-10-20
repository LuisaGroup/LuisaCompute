#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class BasicBlock;

struct PhiIncoming {
    Value *value;
    BasicBlock *block;
};

struct PhiIncomingUse {
    Use *value;
    Use *block;
};

struct ConstPhiIncoming {
    const Value *value;
    const BasicBlock *block;
};

struct ConstPhiIncomingUse {
    const Use *value;
    const Use *block;
};

class LC_XIR_API PhiInst final : public Instruction {
public:
    using Instruction::Instruction;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::PHI;
    }
    void set_incoming_count(size_t count) noexcept;
    void set_incoming(size_t index, Value *value, BasicBlock *block) noexcept;
    void add_incoming(Value *value, BasicBlock *block) noexcept;
    void insert_incoming(size_t index, Value *value, BasicBlock *block) noexcept;
    void remove_incoming(size_t index) noexcept;
    [[nodiscard]] size_t incoming_count() const noexcept;
    [[nodiscard]] PhiIncoming incoming(size_t index) noexcept;
    [[nodiscard]] ConstPhiIncoming incoming(size_t index) const noexcept;
    [[nodiscard]] PhiIncomingUse incoming_use(size_t index) noexcept;
    [[nodiscard]] ConstPhiIncomingUse incoming_use(size_t index) const noexcept;
    [[nodiscard]] luisa::span<PhiIncomingUse> incoming_uses() noexcept;
    [[nodiscard]] luisa::span<const ConstPhiIncomingUse> incoming_uses() const noexcept;
};

}// namespace luisa::compute::xir
