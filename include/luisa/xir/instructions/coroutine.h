#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LUISA_XIR_API CoroRegisterInst final : public DerivedInstruction<CoroRegisterInst, DerivedInstructionTag::CORO_REGISTER> {

public:
    static constexpr size_t operand_index_value = 0u;

private:
    luisa::string _name;

public:
    CoroRegisterInst(BasicBlock *parent_block, Value *value, luisa::string name) noexcept;
    [[nodiscard]] Value *value() noexcept;
    [[nodiscard]] const Value *value() const noexcept;
    [[nodiscard]] auto name() const noexcept { return luisa::string_view{_name}; }
    void set_value(Value *value) noexcept;
    [[nodiscard]] CoroRegisterInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

class LUISA_XIR_API CoroSuspendInst final : public DerivedInstruction<CoroSuspendInst, DerivedInstructionTag::CORO_SUSPEND> {

private:
    uint32_t _token{};

public:
    CoroSuspendInst(BasicBlock *parent_block, uint32_t token) noexcept;
    [[nodiscard]] auto token() const noexcept { return _token; }
    [[nodiscard]] CoroSuspendInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}
