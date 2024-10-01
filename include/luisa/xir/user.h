#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LC_XIR_API User : public Value {

private:
    luisa::vector<Use *> _operands;

public:
    using Value::Value;
    void remove_operand_uses() noexcept;
    void add_operand_uses() noexcept;
    void set_operand(size_t index, Value *value) noexcept;
    [[nodiscard]] Use *operand_use(size_t index) noexcept;
    [[nodiscard]] const Use *operand_use(size_t index) const noexcept;
    [[nodiscard]] Value *operand(size_t index) noexcept;
    [[nodiscard]] const Value *operand(size_t index) const noexcept;
    void set_operand_count(size_t n) noexcept;
    void set_operands(luisa::span<Value *const> operands) noexcept;
    [[nodiscard]] auto operands() noexcept { return luisa::span{_operands}; }
    [[nodiscard]] auto operands() const noexcept { return luisa::span<const Use *const>{_operands}; }
};

}// namespace luisa::compute::xir
