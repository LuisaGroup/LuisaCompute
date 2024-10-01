#pragma once

#include "luisa/xir/basic_block.h"

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class BasicBlock;

// if (cond) { true_block } else { false_block }
class LC_XIR_API BranchInst : public Instruction {

public:
    explicit BranchInst(Pool *pool,
                        Value *cond = nullptr,
                        const Name *name = nullptr) noexcept;

    void set_cond(Value *cond) noexcept;

    [[nodiscard]] Value *cond() noexcept;
    [[nodiscard]] const Value *cond() const noexcept;

    [[nodiscard]] BasicBlock *true_block() noexcept;
    [[nodiscard]] const BasicBlock *true_block() const noexcept;

    [[nodiscard]] BasicBlock *false_block() noexcept;
    [[nodiscard]] const BasicBlock *false_block() const noexcept;
};

}// namespace luisa::compute::xir
