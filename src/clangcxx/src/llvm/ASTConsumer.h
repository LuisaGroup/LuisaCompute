#pragma once
#include <utility>
#include <luisa/vstl/common.h>
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>
#include "luisa/runtime/device.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace luisa::clangcxx {

class ASTConsumer;
using MatchFinder = clang::ast_matchers::MatchFinder;

struct CXXBlackboard
{
    CXXBlackboard();
    ~CXXBlackboard();

    clang::ASTContext* astContext = nullptr;
    luisa::shared_ptr<compute::detail::FunctionBuilder> kernel_builder;
    luisa::unordered_map<luisa::string, const luisa::compute::RefExpr*> globals;
    luisa::unordered_map<const clang::Decl*, luisa::shared_ptr<compute::detail::FunctionBuilder>> func_builders;
    luisa::unordered_map<const clang::Decl*, luisa::shared_ptr<compute::detail::FunctionBuilder>> lambda_builders;
    
    const luisa::compute::Type* RecordAsPrimitiveType(const clang::QualType Type);
    const luisa::compute::Type* RecordAsBuiltinType(const clang::QualType Ty);
    const luisa::compute::Type* RecordAsStuctureType(const clang::QualType Ty);
    const luisa::compute::Type* RecordType(const clang::QualType Ty);

    const luisa::compute::Type* FindOrAddType(const clang::QualType Ty, const clang::ASTContext* astContext);

    luisa::compute::CallOp FindCallOp(const luisa::string_view& name);
    
protected:
    const luisa::compute::Type* findType(const clang::QualType Ty, const clang::ASTContext* astContext);
    bool tryEmplaceFieldType(const clang::QualType Ty, const clang::RecordDecl *decl, luisa::vector<const luisa::compute::Type *> &types);
    bool registerType(clang::QualType Ty, const clang::ASTContext* astContext, const luisa::compute::Type* type);

    vstd::HashMap<vstd::string, luisa::compute::CallOp> ops_map;
    luisa::unordered_map<luisa::string, const luisa::compute::Type*> type_map;
};

struct Stack {
    luisa::unordered_map<luisa::string, const luisa::compute::RefExpr *> locals;
    luisa::unordered_map<const clang::Stmt *, const luisa::compute::Expression *> expr_map;
};

struct FunctionBuilderBuilder
{
    explicit FunctionBuilderBuilder(CXXBlackboard* blackboard, Stack& stack) 
        : blackboard(blackboard), stack(stack) {}
    void build(const clang::FunctionDecl* S);
private:
    bool recursiveVisit(clang::Stmt* stmt, luisa::shared_ptr<compute::detail::FunctionBuilder> cur, Stack& stack);
    CXXBlackboard* blackboard = nullptr;
    Stack& stack;
};

class RecordDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback 
{
public:
    RecordDeclStmtHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;

    CXXBlackboard* blackboard = nullptr;
};

class GlobalVarHandler : public clang::ast_matchers::MatchFinder::MatchCallback 
{
public:
    GlobalVarHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;
    CXXBlackboard* blackboard = nullptr;
};

class FunctionDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback 
{
public:
    FunctionDeclStmtHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;
    CXXBlackboard* blackboard = nullptr;
};

class ASTConsumer : public clang::ASTConsumer
{
public:
    explicit ASTConsumer(std::string OutputPath, luisa::compute::Device* device, compute::ShaderOption option);
    ~ASTConsumer() override;
    void HandleTranslationUnit(clang::ASTContext& Context) override;

    std::string OutputPath;
    luisa::compute::Device* device = nullptr;
    compute::ShaderOption option;

    CXXBlackboard blackboard;

    RecordDeclStmtHandler HandlerForTypeDecl;
    GlobalVarHandler HandlerForGlobalVar;
    FunctionDeclStmtHandler HandlerForFuncionDecl;

    clang::ast_matchers::MatchFinder Matcher;
};

}// namespace lc::clangcxx