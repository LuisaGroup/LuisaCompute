//
// Created by Mike Smith on 2022/5/23.
//

#pragma once

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/IPO.h>

#include <ast/type.h>
#include <ast/expression.h>
#include <ast/statement.h>
#include <ast/function_builder.h>

namespace luisa::compute::llvm {

class LLVMCodegen : public StmtVisitor {

private:
    struct FunctionContext {
        Function function;
        ::llvm::Function *ir;
        ::llvm::Value *ret;
        ::llvm::BasicBlock *exit_block;
        luisa::unique_ptr<::llvm::IRBuilder<>> builder;
        luisa::unordered_map<uint, ::llvm::Value *> variables;
        luisa::vector<::llvm::BasicBlock *> break_targets;
        luisa::vector<::llvm::BasicBlock *> continue_targets;
        luisa::vector<::llvm::SwitchInst *> switch_stack;
        FunctionContext(Function f, ::llvm::Function *ir, ::llvm::Value *ret,
                        ::llvm::BasicBlock *exit_block,
                        luisa::unique_ptr<::llvm::IRBuilder<>> builder,
                        luisa::unordered_map<uint, ::llvm::Value *> variables) noexcept
        : function{f}, ir{ir}, ret{ret}, exit_block{exit_block},
          builder{std::move(builder)}, variables{std::move(variables)} {}
    };

public:
    static constexpr auto buffer_argument_size = 8u;
    static constexpr auto texture_argument_size = 8u;
    static constexpr auto accel_argument_size = 8u;
    static constexpr auto bindless_array_argument_size = 8u;

private:
    struct LLVMStruct {
        ::llvm::StructType *type;
        luisa::vector<uint> member_indices;
    };

private:
    ::llvm::LLVMContext &_context;
    ::llvm::Module *_module{nullptr};
    luisa::unordered_map<uint64_t, LLVMStruct> _struct_types;
    luisa::vector<luisa::unique_ptr<FunctionContext>> _function_stack;

private:
    void _emit_function() noexcept;
    [[nodiscard]] luisa::string _variable_name(Variable v) const noexcept;
    [[nodiscard]] luisa::string _function_name(Function f) const noexcept;
    [[nodiscard]] ::llvm::Function *_create_function(Function f) noexcept;
    [[nodiscard]] luisa::unique_ptr<FunctionContext> _create_kernel_program(Function f) noexcept;
    [[nodiscard]] luisa::unique_ptr<FunctionContext> _create_kernel_context(Function f) noexcept;
    [[nodiscard]] luisa::unique_ptr<FunctionContext> _create_callable_context(Function f) noexcept;
    [[nodiscard]] ::llvm::Type *_create_type(const Type *t) noexcept;
    [[nodiscard]] ::llvm::Value *_create_expr(const Expression *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_unary_expr(const UnaryExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_binary_expr(const BinaryExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_member_expr(const MemberExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_access_expr(const AccessExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_literal_expr(const LiteralExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_ref_expr(const RefExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_constant_expr(const ConstantExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_call_expr(const CallExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_cast_expr(const CastExpr *expr) noexcept;
    [[nodiscard]] ::llvm::Value *_create_stack_variable(::llvm::Value *v, luisa::string_view name = "") noexcept;
    [[nodiscard]] FunctionContext *_current_context() noexcept;
    [[nodiscard]] ::llvm::Value *_convert(const Type *dst_type, const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_bool(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_float(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_int(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_uint(const Type *src_type, ::llvm::Value *p_src) noexcept;
    [[nodiscard]] ::llvm::Value *_scalar_to_vector(const Type *src_type, uint dst_dim, ::llvm::Value *p_src) noexcept;
    void _create_assignment(const Type *dst_type, const Type *src_type, ::llvm::Value *p_dst, ::llvm::Value *p_src) noexcept;

public:
    explicit LLVMCodegen(::llvm::LLVMContext &ctx) noexcept;
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
    void visit(const MetaStmt *stmt) override;
    luisa::unique_ptr<::llvm::Module> emit(Function f) noexcept;
};

}// namespace luisa::compute::llvm
