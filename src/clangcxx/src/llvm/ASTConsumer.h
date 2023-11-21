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

class FunctionDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback 
{
public:
    FunctionDeclStmtHandler() = default;
    bool recursiveVisit(clang::Stmt* stmt, luisa::shared_ptr<compute::detail::FunctionBuilder> cur);
    void run(const MatchFinder::MatchResult &Result) final;
    ASTConsumer* consumer = nullptr;
};

class RecordDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback 
{
public:
    RecordDeclStmtHandler() = default;
    void run(const MatchFinder::MatchResult &Result) final;
    ASTConsumer* consumer = nullptr;
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

    luisa::shared_ptr<compute::detail::FunctionBuilder> kernel_builder;
    const luisa::compute::Type* ttt;

    FunctionDeclStmtHandler HandlerForFuncionDecl;
    RecordDeclStmtHandler HandlerForTypeDecl;
    clang::ASTContext* astContext = nullptr;
    clang::ast_matchers::MatchFinder Matcher;
};

}// namespace lc::clangcxx