#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class BasicBlock;

// do { body } while (cond)
class LC_XIR_API LoopInst : public Instruction {

public:
    explicit LoopInst(Pool *pool, Value *cond = nullptr,
                      const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept final {
        return DerivedInstructionTag::LOOP;
    }

    void set_cond(Value *cond) noexcept;

    [[nodiscard]] Value *cond() noexcept;
    [[nodiscard]] const Value *cond() const noexcept;

    [[nodiscard]] BasicBlock *body() noexcept;
    [[nodiscard]] const BasicBlock *body() const noexcept;
};

}// namespace luisa::compute::xir
