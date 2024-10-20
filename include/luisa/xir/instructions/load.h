#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API LoadInst final : public Instruction {
public:
    explicit LoadInst(Pool *pool, Value *variable = nullptr,
                      const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept final {
        return DerivedInstructionTag::LOAD;
    }
    [[nodiscard]] Value *variable() noexcept;
    [[nodiscard]] const Value *variable() const noexcept;

    void set_variable(Value *variable) noexcept;
};

}// namespace luisa::compute::xir
