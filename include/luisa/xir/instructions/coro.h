#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class LUISA_XIR_API CoroSuspendInst final : public DerivedTerminatorInstruction<CoroSuspendInst, DerivedInstructionTag::CORO_SUSPEND> {

private:
    uint32_t _token;
    luisa::string _name;

public:
    static constexpr size_t operand_index_frame = 0u;

public:
    CoroSuspendInst(BasicBlock *parent_block, uint32_t token, luisa::string name, Value *frame) noexcept;

    [[nodiscard]] auto token() const noexcept { return _token; }
    [[nodiscard]] const luisa::string &name() const noexcept { return _name; }
    [[nodiscard]] auto frame() noexcept { return operand(operand_index_frame); }
    [[nodiscard]] auto frame() const noexcept { return operand(operand_index_frame); }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) noexcept {
        return visitor(*this);
    }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) const noexcept {
        return visitor(*this);
    }

    [[nodiscard]] CoroSuspendInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

class LUISA_XIR_API CoroResumeInst final : public DerivedInstruction<CoroResumeInst, DerivedInstructionTag::CORO_RESUME> {

private:
    uint32_t _token;

public:
    static constexpr size_t operand_index_frame = 0u;

public:
    CoroResumeInst(BasicBlock *parent_block, uint32_t token, Value *frame) noexcept;

    [[nodiscard]] auto token() const noexcept { return _token; }
    [[nodiscard]] auto frame() noexcept { return operand(operand_index_frame); }
    [[nodiscard]] auto frame() const noexcept { return operand(operand_index_frame); }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) noexcept {
        return visitor(*this);
    }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) const noexcept {
        return visitor(*this);
    }

    [[nodiscard]] CoroResumeInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

class LUISA_XIR_API CoroTerminateInst final : public DerivedTerminatorInstruction<CoroTerminateInst, DerivedInstructionTag::CORO_TERMINATE> {

public:
    explicit CoroTerminateInst(BasicBlock *parent_block) noexcept;

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) noexcept {
        return visitor(*this);
    }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) const noexcept {
        return visitor(*this);
    }

    [[nodiscard]] CoroTerminateInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

class LUISA_XIR_API CoroRegisterInst final : public DerivedInstruction<CoroRegisterInst, DerivedInstructionTag::CORO_REGISTER> {

private:
    luisa::string _name;

public:
    static constexpr size_t operand_index_value = 0u;
    static constexpr size_t operand_index_frame = 1u;

public:
    CoroRegisterInst(BasicBlock *parent_block, luisa::string name, Value *value, Value *frame) noexcept;

    [[nodiscard]] const luisa::string &name() const noexcept { return _name; }
    [[nodiscard]] auto value() noexcept { return operand(operand_index_value); }
    [[nodiscard]] auto value() const noexcept { return operand(operand_index_value); }
    [[nodiscard]] auto frame() noexcept { return operand(operand_index_frame); }
    [[nodiscard]] auto frame() const noexcept { return operand(operand_index_frame); }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) noexcept {
        return visitor(*this);
    }

    template<typename Visitor>
    decltype(auto) accept(Visitor &&visitor) const noexcept {
        return visitor(*this);
    }

    [[nodiscard]] CoroRegisterInst *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
