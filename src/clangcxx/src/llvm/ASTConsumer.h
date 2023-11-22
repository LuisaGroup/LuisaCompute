#pragma once
#include <utility>
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>
#include "luisa/core/dll_export.h"
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
    clang::ASTContext* astContext = nullptr;
    luisa::shared_ptr<compute::detail::FunctionBuilder> kernel_builder;
    luisa::unordered_map<luisa::string, const luisa::compute::Type*> type_map;
};

class FunctionDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback 
{
public:
    FunctionDeclStmtHandler() = default;
    bool recursiveVisit(clang::Stmt* stmt, luisa::shared_ptr<compute::detail::FunctionBuilder> cur);
    void run(const MatchFinder::MatchResult &Result) final;
    CXXBlackboard* blackboard = nullptr;
};

class RecordDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback 
{
public:
    RecordDeclStmtHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;
    bool TryEmplaceAsPrimitiveType(const clang::BuiltinType *builtin, luisa::vector<const luisa::compute::Type *> &types);
    bool TryEmplaceAsBuiltinType(const clang::QualType Ty, const clang::RecordDecl* recordDecl, luisa::vector<const luisa::compute::Type *> &types);
    bool TryEmplaceAsStructureType(const clang::RecordDecl* recordDecl, luisa::vector<const luisa::compute::Type *> &types);

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

    FunctionDeclStmtHandler HandlerForFuncionDecl;
    RecordDeclStmtHandler HandlerForTypeDecl;
    clang::ast_matchers::MatchFinder Matcher;
};

}// namespace lc::clangcxx