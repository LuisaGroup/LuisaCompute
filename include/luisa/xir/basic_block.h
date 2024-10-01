#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class Function;
class Instruction;

class LC_XIR_API BasicBlock : public Value {

private:
    InlineInstructionList _instructions;

public:
    explicit BasicBlock(Pool *pool, const Name *name = nullptr) noexcept;
    [[nodiscard]] Instruction *parent_instruction() noexcept;
    [[nodiscard]] const Instruction *parent_instruction() const noexcept;
    [[nodiscard]] BasicBlock *parent_block() noexcept;
    [[nodiscard]] const BasicBlock *parent_block() const noexcept;
    [[nodiscard]] auto &instructions() noexcept { return _instructions; }
    [[nodiscard]] auto &instructions() const noexcept { return _instructions; }
};

}// namespace luisa::compute::xir
