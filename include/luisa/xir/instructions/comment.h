#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API CommentInst final : public Instruction {

private:
    luisa::string _comment;

public:
    explicit CommentInst(Pool *pool, luisa::string comment = {},
                         const Name *name = nullptr) noexcept;

    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::COMMENT;
    }

    void set_comment(luisa::string_view comment) noexcept;
    [[nodiscard]] auto comment() const noexcept { return luisa::string_view{_comment}; }
};

}// namespace luisa::compute::xir
