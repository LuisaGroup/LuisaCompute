#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LC_XIR_API User : public Value {

private:
    luisa::vector<Use *> _operands;

protected:
    void _add_operand_uses() noexcept;
    void _remove_operand_uses() noexcept;

public:
    using Value::Value;

    void set_operand_count(size_t n) noexcept;
    void set_operand(size_t index, Value *value) noexcept;
    void set_operands(luisa::span<Value *const> operands) noexcept;

    void add_operand(Value *value) noexcept;
    void insert_operand(size_t index, Value *value) noexcept;
    void remove_operand(size_t index) noexcept;

    [[nodiscard]] Use *operand_use(size_t index) noexcept;
    [[nodiscard]] const Use *operand_use(size_t index) const noexcept;

    [[nodiscard]] Value *operand(size_t index) noexcept;
    [[nodiscard]] const Value *operand(size_t index) const noexcept;

    [[nodiscard]] luisa::span<Use *> operand_uses() noexcept;
    [[nodiscard]] luisa::span<const Use *const> operand_uses() const noexcept;

    [[nodiscard]] size_t operand_count() const noexcept;
};

}// namespace luisa::compute::xir
