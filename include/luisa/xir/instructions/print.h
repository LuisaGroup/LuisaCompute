#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API PrintInst final : public Instruction {

private:
    luisa::string _format;

public:
    explicit PrintInst(Pool *pool, luisa::string format = {},
                       luisa::span<Value *const> operands = {},
                       const Name *name = nullptr) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::PRINT;
    }
    [[nodiscard]] auto format() const noexcept { return luisa::string_view{_format}; }
    void set_format(luisa::string_view format) noexcept { _format = format; }
};

}// namespace luisa::compute::xir
