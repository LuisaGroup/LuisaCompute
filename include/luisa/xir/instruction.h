#pragma once

#include <luisa/xir/user.h>

namespace luisa::compute::xir {

class BasicBlock;

class LC_XIR_API Instruction : public IntrusiveNode<Instruction, User> {

private:
    BasicBlock *_parent_block = nullptr;

public:
    explicit Instruction(Pool *pool,
                         const Type *type = nullptr,
                         BasicBlock *parent_block = nullptr,
                         const Name *name = nullptr) noexcept;
    void remove_self() noexcept override;
    void insert_before_self(Instruction *node) noexcept override;
    void insert_after_self(Instruction *node) noexcept override;
    void set_parent_block(BasicBlock *block) noexcept { _parent_block = block; }
    [[nodiscard]] BasicBlock *parent_block() noexcept { return _parent_block; }
    [[nodiscard]] const BasicBlock *parent_block() const noexcept { return _parent_block; }
};

using InlineInstructionList = InlineIntrusiveList<Instruction>;

}// namespace luisa::compute::xir
