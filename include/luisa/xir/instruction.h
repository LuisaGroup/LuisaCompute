#pragma once

#include <luisa/xir/user.h>

namespace luisa::compute::xir {

class BasicBlock;

enum struct DerivedInstructionTag {
    SENTINEL,
    BRANCH,
    COMMENT,
    LOOP,
};

class LC_XIR_API Instruction : public IntrusiveNode<Instruction, User> {

private:
    friend BasicBlock;
    BasicBlock *_parent_block = nullptr;

protected:
    void _set_parent_block(BasicBlock *block) noexcept;

public:
    explicit Instruction(Pool *pool, const Type *type = nullptr,
                         const Name *name = nullptr) noexcept;
    [[nodiscard]] virtual DerivedInstructionTag derived_instruction_tag() const noexcept {
        return DerivedInstructionTag::SENTINEL;
    }
    [[nodiscard]] DerivedValueTag derived_value_tag() const noexcept final {
        return DerivedValueTag::INSTRUCTION;
    }

    void remove_self() noexcept override;
    void insert_before_self(Instruction *node) noexcept override;
    void insert_after_self(Instruction *node) noexcept override;
    [[nodiscard]] BasicBlock *parent_block() noexcept { return _parent_block; }
    [[nodiscard]] const BasicBlock *parent_block() const noexcept { return _parent_block; }
};

using InstructionList = InlineIntrusiveList<Instruction>;

}// namespace luisa::compute::xir
