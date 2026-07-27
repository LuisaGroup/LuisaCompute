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
    enum class Kind { UNKNOWN,
                      CONSTANT,
                      ADD_REC,
                      ADD,
                      MUL };
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
    size_t rejected_loop_count{0u};
    size_t invalid_function_count{0u};

    [[nodiscard]] bool succeeded() const noexcept {
        return rejected_loop_count == 0u && invalid_function_count == 0u;
    }
};

// Null inputs and malformed bodyless kernels fail with
// invalid_function_count. A bodyless callable is an external declaration and
// produces a successful empty analysis.
[[nodiscard]] LUISA_XIR_API SCEVInfo scev_pass_run_on_function(FunctionDefinition *def) noexcept;
[[nodiscard]] LUISA_XIR_API SCEVInfo scev_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API const SCEV *scev_get_for_value(Instruction *inst) noexcept;

class LUISA_XIR_API SCEVAnalysis {
private:
    struct Impl;
    luisa::unique_ptr<Impl> _impl;
    [[nodiscard]] const SCEV *_get_unchecked(Instruction *inst) const noexcept;
    friend LUISA_XIR_API SCEVInfo scev_pass_run_on_function(FunctionDefinition *def) noexcept;
    friend LUISA_XIR_API const SCEV *scev_get_for_value(Instruction *inst) noexcept;

public:
    SCEVAnalysis() noexcept;
    ~SCEVAnalysis() noexcept;
    SCEVAnalysis(SCEVAnalysis &&) noexcept;
    SCEVAnalysis &operator=(SCEVAnalysis &&) noexcept;
    SCEVAnalysis(const SCEVAnalysis &) = delete;
    SCEVAnalysis &operator=(const SCEVAnalysis &) = delete;

    void clear() noexcept;
    [[nodiscard]] SCEVInfo analyze(FunctionDefinition *def) noexcept;
    [[nodiscard]] const SCEV *get(Instruction *inst) const noexcept;
    [[nodiscard]] FunctionDefinition *function() const noexcept;
    [[nodiscard]] bool is_current() const noexcept;
};

namespace detail {
LUISA_XIR_API void scev_register_function(Function *function) noexcept;
LUISA_XIR_API void scev_invalidate_function(Function *function) noexcept;
}// namespace detail

}// namespace luisa::compute::xir
