#pragma once

#include "luisa/xir/basic_block.h"

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class BasicBlock;

class LC_XIR_API BranchInst : public Instruction {

public:
    explicit BranchInst(Pool *pool,
                        Value *cond = nullptr,
                        BasicBlock *true_block = nullptr,
                        BasicBlock *false_block = nullptr,
                        BasicBlock *parent_block = nullptr,
                        const Name *name = nullptr) noexcept;

    void set_cond(Value *cond) noexcept;
    void set_true_block(BasicBlock *block) noexcept;
    void set_false_block(BasicBlock *block) noexcept;

    [[nodiscard]] Value *cond() noexcept;
    [[nodiscard]] const Value *cond() const noexcept;

    [[nodiscard]] BasicBlock *true_block() noexcept;
    [[nodiscard]] const BasicBlock *true_block() const noexcept;

    [[nodiscard]] BasicBlock *false_block() noexcept;
    [[nodiscard]] const BasicBlock *false_block() const noexcept;
};

}// namespace luisa::compute::xir
