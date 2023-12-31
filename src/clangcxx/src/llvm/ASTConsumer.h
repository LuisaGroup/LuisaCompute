#pragma once
#include "Utils/AttributeHelper.hpp"
#include "TypeDatabase.h"

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/variant.h>
#include <luisa/runtime/device.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/ast/callable_library.h>

#include <clang/AST/Decl.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecordLayout.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>

namespace luisa::clangcxx {

class ASTConsumer;
using MatchFinder = clang::ast_matchers::MatchFinder;

struct Stack {
    const luisa::compute::RefExpr *GetLocal(const clang::ValueDecl *decl) const;
    void SetLocal(const clang::ValueDecl *decl, const luisa::compute::RefExpr *expr);

    const luisa::compute::Expression *GetExpr(const clang::Stmt *stmt) const;
    void SetExpr(const clang::Stmt *stmt, const luisa::compute::Expression *expr);

    const luisa::compute::Expression *GetConstant(const clang::ValueDecl *var) const;
    void SetConstant(const clang::ValueDecl *var, const luisa::compute::Expression *expr);

    bool isCtorExpr(const luisa::compute::Expression *expr);
    void SetExprAsCtor(const luisa::compute::Expression *expr);

    luisa::vector<const luisa::compute::Expression *> callers;

private:
    luisa::unordered_set<const luisa::compute::Expression *> ctor_exprs;
    luisa::unordered_map<const clang::Stmt *, const luisa::compute::Expression *> expr_map;
    luisa::unordered_map<const clang::ValueDecl *, const luisa::compute::RefExpr *> locals;
    luisa::unordered_map<const clang::ValueDecl *, const luisa::compute::Expression *> constants;
};

struct FunctionBuilderBuilder {
    explicit FunctionBuilderBuilder(TypeDatabase *db, Stack &stack)
        : db(db), stack(stack) {}
    // return kernel dimension, 0 if not kernel
    struct BuildResult {
        compute::Function func;
        uint dimension;
    };
    BuildResult build(const clang::FunctionDecl *S, bool allowKernel);

private:
    bool recursiveVisit(clang::Stmt *stmt, compute::detail::FunctionBuilder *cur, Stack &stack);
    TypeDatabase *db = nullptr;
    Stack &stack;
};

struct RecordDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback {
    RecordDeclStmtHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;

    TypeDatabase *db = nullptr;
};

struct GlobalVarHandler : public clang::ast_matchers::MatchFinder::MatchCallback {
    GlobalVarHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;

    TypeDatabase *db = nullptr;
};

struct FunctionDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback {
    FunctionDeclStmtHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;
    uint dimension = 0;
    TypeDatabase *db = nullptr;
    compute::CallableLibrary *call_lib = nullptr;
};

class ASTConsumerBase : public clang::ASTConsumer {
public:
    explicit ASTConsumerBase();
    void HandleTranslationUnit(clang::ASTContext &Context) override;
    virtual ~ASTConsumerBase() override;
protected:
    TypeDatabase db;
    RecordDeclStmtHandler HandlerForTypeDecl;
    GlobalVarHandler HandlerForGlobalVar;
    FunctionDeclStmtHandler HandlerForFuncionDecl;
    clang::ast_matchers::MatchFinder Matcher;
};

class ASTConsumer final : public ASTConsumerBase {
public:
    explicit ASTConsumer(luisa::compute::Device *device, compute::ShaderOption option);
    ~ASTConsumer() override;
    const luisa::compute::Device *device = nullptr;
    const compute::ShaderOption option;
};
class ASTCallableConsumer final : public ASTConsumerBase {
public:
    explicit ASTCallableConsumer(compute::CallableLibrary* lib);
    ~ASTCallableConsumer() override;
};
}// namespace luisa::clangcxx