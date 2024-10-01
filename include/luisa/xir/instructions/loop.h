#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class BasicBlock;

// do { body } while (cond)
class LoopInst : public Instruction {

public:
    explicit LoopInst(Pool *pool,
                      Value *cond = nullptr,
                      const Name *name = nullptr) noexcept;

    void set_cond(Value *cond) noexcept;

    [[nodiscard]] Value *cond() noexcept;
    [[nodiscard]] const Value *cond() const noexcept;

    [[nodiscard]] BasicBlock *body() noexcept;
    [[nodiscard]] const BasicBlock *body() const noexcept;
};

}// namespace luisa::compute::xir
