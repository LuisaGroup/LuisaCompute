#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class Constant;
class LoopInst;
class PassReport;
class Value;

class SCEV {
public:
    enum class Kind { UNKNOWN, CONSTANT, ADD_REC, ADD, MUL };
    virtual ~SCEV() = default;
    [[nodiscard]] virtual Kind kind() const noexcept = 0;
    [[nodiscard]] virtual const Type *type() const noexcept = 0;
};

class SCEVUnknown : public SCEV {
    Value *_value;

public:
    explicit SCEVUnknown(Value *value) noexcept;
    [[nodiscard]] Kind kind() const noexcept override { return Kind::UNKNOWN; }
    [[nodiscard]] const Type *type() const noexcept override;
    [[nodiscard]] Value *value() const noexcept { return _value; }
    [[nodiscard]] Instruction *inst() const noexcept {
        return _value != nullptr && _value->isa<Instruction>() ?
                   static_cast<Instruction *>(_value) :
                   nullptr;
    }
};

class SCEVConstant : public SCEV {
    Constant *_constant;

public:
    explicit SCEVConstant(Constant *c) noexcept;
    [[nodiscard]] Kind kind() const noexcept override { return Kind::CONSTANT; }
    [[nodiscard]] const Type *type() const noexcept override;
    [[nodiscard]] Constant *constant() const noexcept { return _constant; }
};

class SCEVAddRec : public SCEV {
    const SCEV *_start;
    const SCEV *_stride;
    LoopInst *_loop;

public:
    SCEVAddRec(const SCEV *start, const SCEV *stride, LoopInst *loop) noexcept;
    [[nodiscard]] Kind kind() const noexcept override { return Kind::ADD_REC; }
    [[nodiscard]] const Type *type() const noexcept override;
    [[nodiscard]] const SCEV *start() const noexcept { return _start; }
    [[nodiscard]] const SCEV *stride() const noexcept { return _stride; }
    [[nodiscard]] LoopInst *loop() const noexcept { return _loop; }
};

class SCEVAddExpr : public SCEV {
    luisa::vector<const SCEV *> _operands;

public:
    explicit SCEVAddExpr(luisa::vector<const SCEV *> ops) noexcept;
    [[nodiscard]] Kind kind() const noexcept override { return Kind::ADD; }
    [[nodiscard]] const Type *type() const noexcept override;
    [[nodiscard]] luisa::span<const SCEV *const> operands() const noexcept { return _operands; }
};

class SCEVMulExpr : public SCEV {
    luisa::vector<const SCEV *> _operands;

public:
    explicit SCEVMulExpr(luisa::vector<const SCEV *> ops) noexcept;
    [[nodiscard]] Kind kind() const noexcept override { return Kind::MUL; }
    [[nodiscard]] const Type *type() const noexcept override;
    [[nodiscard]] luisa::span<const SCEV *const> operands() const noexcept { return _operands; }
};

struct SCEVInfo {
    size_t analyzed_loop_count{0u};
};

[[nodiscard]] LUISA_XIR_API SCEVInfo scev_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API SCEVInfo scev_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API const SCEV *scev_get_for_value(Instruction *inst) noexcept;

}// namespace luisa::compute::xir
