#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class Function;
class Instruction;

class LC_XIR_API BasicBlock : public Value {

private:
    Value *_parent_value = nullptr;
    InstructionList _instructions;

public:
    explicit BasicBlock(Pool *pool, Value *parent_value = nullptr,
                        const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedValueTag derived_value_tag() const noexcept final {
        return DerivedValueTag::BASIC_BLOCK;
    }

    void set_parent_value(Value *parent_value) noexcept;
    [[nodiscard]] Value *parent_value() noexcept { return _parent_value; }
    [[nodiscard]] const Value *parent_value() const noexcept { return _parent_value; }
    [[nodiscard]] auto &instructions() noexcept { return _instructions; }
    [[nodiscard]] auto &instructions() const noexcept { return _instructions; }
};

}// namespace luisa::compute::xir
