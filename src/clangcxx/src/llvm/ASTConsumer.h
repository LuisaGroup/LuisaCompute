#pragma once
#include "Utils/AttributeHelper.hpp"
#include "TypeDatabase.h"

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/variant.h>
#include <luisa/runtime/device.h>
#include <luisa/dsl/rtx/ray_query.h>

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
    const luisa::compute::RefExpr* GetLocal(const clang::ValueDecl *decl) const;
    void SetLocal(const clang::ValueDecl *decl, const luisa::compute::RefExpr *expr);

    const luisa::compute::Expression* GetExpr(const clang::Stmt *stmt) const;
    void SetExpr(const clang::Stmt *stmt, const luisa::compute::Expression *expr);

    luisa::vector<const luisa::compute::Expression *> callers;
    luisa::vector<class luisa::compute::RayQueryStmt*> queries;

private:
    luisa::unordered_map<const clang::Stmt *, const luisa::compute::Expression *> expr_map;
    luisa::unordered_map<const clang::ValueDecl *, const luisa::compute::RefExpr *> locals;
};

struct FunctionBuilderBuilder {
    explicit FunctionBuilderBuilder(TypeDatabase *db, Stack &stack)
        : db(db), stack(stack) {}
    void build(const clang::FunctionDecl *S);
private:
    bool recursiveVisit(clang::Stmt *stmt, luisa::shared_ptr<compute::detail::FunctionBuilder> cur, Stack &stack);
    TypeDatabase *db = nullptr;
    Stack &stack;
};

class RecordDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
    RecordDeclStmtHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;

    TypeDatabase *db = nullptr;
};

class GlobalVarHandler : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
    GlobalVarHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;
    TypeDatabase *db = nullptr;
};

class FunctionDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
    FunctionDeclStmtHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;
    TypeDatabase *db = nullptr;
};

class ASTConsumer : public clang::ASTConsumer {
public:
    explicit ASTConsumer(std::string OutputPath, luisa::compute::Device *device, compute::ShaderOption option);
    ~ASTConsumer() override;
    void HandleTranslationUnit(clang::ASTContext &Context) override;

    std::string OutputPath;
    luisa::compute::Device *device = nullptr;
    compute::ShaderOption option;

    TypeDatabase db;

    RecordDeclStmtHandler HandlerForTypeDecl;
    GlobalVarHandler HandlerForGlobalVar;
    FunctionDeclStmtHandler HandlerForFuncionDecl;

    clang::ast_matchers::MatchFinder Matcher;
};

}// namespace luisa::clangcxx