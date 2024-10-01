#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class Function;
class Instruction;

class LC_XIR_API BasicBlock : public Value {

private:
    Function *_function = nullptr;
    InlineInstructionList _instructions;

public:
    explicit BasicBlock(Pool *pool,
                        Function *function = nullptr,
                        const Name *name = nullptr) noexcept;
    void set_function(Function *function) noexcept;
    [[nodiscard]] auto function() noexcept { return _function; }
    [[nodiscard]] auto function() const noexcept { return const_cast<const Function *>(_function); }
    [[nodiscard]] auto &instructions() noexcept { return _instructions; }
    [[nodiscard]] auto &instructions() const noexcept { return _instructions; }
};

}// namespace luisa::compute::xir
