#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class Function;
class Instruction;

class LC_XIR_API BasicBlock : public Value {

private:
    InstructionList _instructions;

public:
    explicit BasicBlock(Pool *pool, const Name *name = nullptr) noexcept;
    [[nodiscard]] virtual bool is_function_body() const noexcept { return false; }
    [[nodiscard]] DerivedValueTag derived_value_tag() const noexcept final {
        return DerivedValueTag::BASIC_BLOCK;
    }

    [[nodiscard]] Instruction *parent_instruction() noexcept;
    [[nodiscard]] const Instruction *parent_instruction() const noexcept;
    [[nodiscard]] BasicBlock *parent_block() noexcept;
    [[nodiscard]] const BasicBlock *parent_block() const noexcept;
    [[nodiscard]] auto &instructions() noexcept { return _instructions; }
    [[nodiscard]] auto &instructions() const noexcept { return _instructions; }
};

class LC_XIR_API FunctionBodyBlock : public BasicBlock {

private:
    Function *_parent_function;

public:
    explicit FunctionBodyBlock(Pool *pool,
                               Function *parent_function = nullptr,
                               const Name *name = nullptr) noexcept;
    [[nodiscard]] bool is_function_body() const noexcept override { return true; }
    void set_parent_function(Function *function) noexcept;
    [[nodiscard]] Function *parent_function() noexcept { return _parent_function; }
    [[nodiscard]] const Function *parent_function() const noexcept { return _parent_function; }
};

}// namespace luisa::compute::xir
