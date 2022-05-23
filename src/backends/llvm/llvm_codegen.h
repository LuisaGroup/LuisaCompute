//
// Created by Mike Smith on 2022/5/23.
//

#pragma once

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
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

class LLVMCodegen : public ExprVisitor, public StmtVisitor {

private:
    struct LLVMStruct {
        ::llvm::StructType *type;
        luisa::vector<uint> member_indices;
    };

private:
    ::llvm::LLVMContext &_context;
    ::llvm::Module *_module{nullptr};
    luisa::unique_ptr<::llvm::IRBuilder<>> _builder;
    luisa::unordered_map<luisa::string, ::llvm::Value *> _values;
    luisa::unordered_map<uint64_t, LLVMStruct> _struct_types;
    Function _function;

private:
    ::llvm::Function *_emit(Function f) noexcept;
    ::llvm::FunctionType *_create_function_type(Function f) noexcept;
    ::llvm::Type *_create_type(const Type *t) noexcept;

public:
    explicit LLVMCodegen(::llvm::LLVMContext &ctx) noexcept;
    void visit(const UnaryExpr *expr) override;
    void visit(const BinaryExpr *expr) override;
    void visit(const MemberExpr *expr) override;
    void visit(const AccessExpr *expr) override;
    void visit(const LiteralExpr *expr) override;
    void visit(const RefExpr *expr) override;
    void visit(const ConstantExpr *expr) override;
    void visit(const CallExpr *expr) override;
    void visit(const CastExpr *expr) override;
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
    void visit(const Type *type) noexcept override;
    luisa::unique_ptr<::llvm::Module> emit(Function f) noexcept;
};

}// namespace luisa::compute::llvm
