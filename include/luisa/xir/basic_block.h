#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class Function;
class Instruction;

class LC_XIR_API BasicBlock : public PooledObject {

private:
    Function *_function = nullptr;
    BasicBlock *_parent_block = nullptr;
    const Name *_name = nullptr;
    InlineInstructionList _instructions;

public:
    explicit BasicBlock(Function *function = nullptr,
                        BasicBlock *parent_block = nullptr,
                        const Name *name = nullptr) noexcept;
    void set_function(Function *function) noexcept;
    void set_parent_block(BasicBlock *parent_block) noexcept;
    void set_name(const Name *name) noexcept;
    [[nodiscard]] auto function() const noexcept { return _function; }
    [[nodiscard]] auto parent_block() const noexcept { return _parent_block; }
    [[nodiscard]] auto name() const noexcept { return _name; }
    [[nodiscard]] auto &instructions() noexcept { return _instructions; }
    [[nodiscard]] auto &instructions() const noexcept { return _instructions; }
};

}// namespace luisa::compute::xir
