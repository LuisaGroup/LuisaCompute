#pragma once

#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/ast/function.h>
#include <luisa/ast/expression.h>
#include <luisa/ast/statement.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/logging.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>

#include <memory>
#include <string>

namespace lc::llvm_codegen {

using namespace luisa;
using namespace luisa::compute;

class LLVMCodegenUtility;
struct LLVMCodegenStackData;

/**
 * @brief LLVM IR state visitor — walks the AST and builds LLVM IR.
 *
 * Inherits StmtVisitor + ExprVisitor and mirrors hlsl::StringStateVisitor.
 * Expression visitors return llvm::Value* (stored in a member for the visitor pattern).
 */
class LLVMStateVisitor final : public StmtVisitor, public ExprVisitor {
public:
    Function f;

private:
    LLVMCodegenUtility *_util;
    llvm::LLVMContext &_ctx;
    llvm::Module &_module;
    llvm::IRBuilder<> &_builder;

    // Scratch value: expression visitors set this
    llvm::Value *_last_value{nullptr};

    // Entry basic block (for alloca placement)
    llvm::BasicBlock *_entry_block{nullptr};

    // Switch state
    llvm::SwitchInst *_current_switch{nullptr};
    llvm::BasicBlock *_switch_merge_block{nullptr};

public:
    // --- Expression visitors (return void, set _last_value) ---
    void visit(const UnaryExpr *expr) override;
    void visit(const BinaryExpr *expr) override;
    void visit(const MemberExpr *expr) override;
    void visit(const AccessExpr *expr) override;
    void visit(const LiteralExpr *expr) override;
    void visit(const RefExpr *expr) override;
    void visit(const CallExpr *expr) override;
    void visit(const CastExpr *expr) override;
    void visit(const ConstantExpr *expr) override;
    void visit(const TypeIDExpr *expr) override;
    void visit(const StringIDExpr *expr) override;
    void visit(const FuncRefExpr *expr) override { LUISA_NOT_IMPLEMENTED(); }
    void visit(const CpuCustomOpExpr *expr) override { LUISA_NOT_IMPLEMENTED(); }
    void visit(const GpuCustomOpExpr *expr) override { LUISA_NOT_IMPLEMENTED(); }

    // --- Statement visitors ---
    void visit(const BreakStmt *stmt) override;
    void visit(const ContinueStmt *stmt) override;
    void visit(const ReturnStmt *stmt) override;
    void visit(const ScopeStmt *stmt) override;
    void visit(const IfStmt *stmt) override;
    void visit(const LoopStmt *stmt) override;
    void visit(const ExprStmt *stmt) override;
    void visit(const SwitchStmt *stmt) override;
    void visit(const SwitchCaseStmt *stmt) override;
    void visit(const SwitchDefaultStmt *stmt) override;
    void visit(const AssignStmt *stmt) override;
    void visit(const ForStmt *stmt) override;
    void visit(const CommentStmt *stmt) override;
    void visit(const RayQueryStmt *stmt) override;
    void visit(const AutoDiffStmt *stmt) override;
    void visit(const PrintStmt *stmt) override;
    void visit(const DebugBreakStmt *stmt) override;

    // --- Main entry ---
    void VisitFunction(Function func);

    // --- Helpers ---
    [[nodiscard]] llvm::Value *EvalExpr(Expression const *expr);
    [[nodiscard]] llvm::Type *ToLLVMType(Type const &type);
    [[nodiscard]] llvm::Value *GetVariable(uint32_t uid, Type const *type);
    void StoreVariable(uint32_t uid, llvm::Value *value);

    LLVMStateVisitor(Function f, LLVMCodegenUtility &util);
    ~LLVMStateVisitor() = default;

private:
    void _push_loop(llvm::BasicBlock *break_target, llvm::BasicBlock *continue_target);
    void _pop_loop();
    void _codegen_builtin_call(CallOp op, const CallExpr *expr);

    // Math helpers
    [[nodiscard]] llvm::Value *_emit_abs(llvm::Value *v, Type const &type);
    [[nodiscard]] llvm::Value *_emit_min(llvm::Value *a, llvm::Value *b, Type const &type);
    [[nodiscard]] llvm::Value *_emit_max(llvm::Value *a, llvm::Value *b, Type const &type);
    [[nodiscard]] llvm::Value *_emit_clamp(llvm::Value *v, llvm::Value *lo, llvm::Value *hi, Type const &type);
    [[nodiscard]] llvm::Value *_emit_lerp(llvm::Value *a, llvm::Value *b, llvm::Value *t, Type const &type);
    [[nodiscard]] llvm::Value *_emit_length(llvm::Value *v);
    [[nodiscard]] llvm::Value *_emit_normalize(llvm::Value *v);
    [[nodiscard]] llvm::Value *_emit_dot(llvm::Value *a, llvm::Value *b);
    [[nodiscard]] llvm::Value *_emit_cross(llvm::Value *a, llvm::Value *b);
    [[nodiscard]] llvm::Value *_emit_all(llvm::Value *v);
    [[nodiscard]] llvm::Value *_emit_any(llvm::Value *v);
};

} // namespace lc::llvm_codegen
