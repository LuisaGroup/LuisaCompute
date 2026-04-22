#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LUISA_XIR_API CoroRegisterInst final : public PrintMessageMixin<DerivedInstruction<CoroRegisterInst, DerivedInstructionTag::CORO_REGISTER>> {
public:
    static constexpr size_t operand_index_value = 0u;

public:
    CoroRegisterInst(BasicBlock *parent_block, Value *value, luisa::string name) noexcept;
    [[nodiscard]] Value *value() noexcept;
    [[nodiscard]] const Value *value() const noexcept;
    void set_value(Value *value) noexcept;
    [[nodiscard]] luisa::string_view name() const noexcept { return message(); }
    void set_name(luisa::string_view name) noexcept { set_message(name); }
    [[nodiscard]] CoroRegisterInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
