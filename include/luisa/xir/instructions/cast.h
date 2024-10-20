#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

enum struct CastOp {
    STATIC_CAST,
    BITWISE_CAST,
};

class LC_XIR_API CastInst final : public Instruction {

private:
    CastOp _op;

public:
    explicit CastInst(Pool *pool, CastOp op = CastOp::STATIC_CAST,
                      Value *value = nullptr,
                      const Type *target_type = nullptr,
                      const Name *name = nullptr) noexcept;

    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override {
        return DerivedInstructionTag::CAST;
    }

    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] Value *value() noexcept;
    [[nodiscard]] const Value *value() const noexcept;

    void set_op(CastOp op) noexcept;
    void set_value(Value *value) noexcept;
};

}// namespace luisa::compute::xir
