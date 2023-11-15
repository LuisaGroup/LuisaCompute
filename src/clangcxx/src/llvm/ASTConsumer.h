#pragma once
#include <utility>
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace luisa::clangcxx {

using MatchFinder = clang::ast_matchers::MatchFinder;

class FunctionDeclStmtHandler : public clang::ast_matchers::MatchFinder::MatchCallback 
{
public:
    FunctionDeclStmtHandler() = default;
    bool recursiveVisit(clang::Stmt* stmt);
    void run(const MatchFinder::MatchResult &Result) final;
    
    llvm::ArrayRef<clang::ParmVarDecl*> params;
};

class ASTConsumer : public clang::ASTConsumer
{
public:
    explicit ASTConsumer(std::string OutputPath);
    void HandleTranslationUnit(clang::ASTContext& Context) override;

    std::string OutputPath;
    FunctionDeclStmtHandler HandlerForFuncionDecl;
    clang::ASTContext* astContext = nullptr;
    clang::ast_matchers::MatchFinder Matcher;
};

}// namespace lc::clangcxx