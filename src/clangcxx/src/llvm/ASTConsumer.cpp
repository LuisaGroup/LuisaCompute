#include "ASTConsumer.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include "AttributeHelpers.hpp"
#include <iostream>

namespace luisa::clangcxx {

using namespace clang;
using namespace clang::ast_matchers;

bool FunctionDeclStmtHandler::recursiveVisit(clang::Stmt *stmt) {
    if (!stmt)
        return true;

    for (Stmt::child_iterator i = stmt->child_begin(), e = stmt->child_end(); i != e; ++i) {
        Stmt *currStmt = *i;
        if (!currStmt)
            continue;
        if (isa<clang::DeclRefExpr>(currStmt)) {
            auto* declRef = (DeclRefExpr *)currStmt;
            bool found = false;
            for (unsigned int paramIndex = 0; paramIndex < params.size(); paramIndex++) 
            {

            }
        }

        currStmt->dump();
        recursiveVisit(currStmt);
    }
    return true;
}

void FunctionDeclStmtHandler::run(const MatchFinder::MatchResult &Result) {
    // The matched 'if' statement was bound to 'ifStmt'.
    if (const auto *S = Result.Nodes.getNodeAs<clang::FunctionDecl>("FunctionDecl")) {
        // S->dump();
        bool ignore = false;
        params = S->parameters();
        if (auto Anno = S->getAttr<clang::AnnotateAttr>())
        {
            ignore = isIgnore(Anno);
        }
        if (!ignore)
        {
            std::cout << S->getName().data() << std::endl;
    
            Stmt *body = S->getBody();
            recursiveVisit(body);
        }
    }
}

ASTConsumer::ASTConsumer(std::string OutputPath)
    : OutputPath(std::move(OutputPath)) {

    Matcher.addMatcher(functionDecl(
                           isDefinition(),
                           unless(isExpansionInSystemHeader()))
                           .bind("FunctionDecl"),
                       &HandlerForFuncionDecl);
    // Matcher.addMatcher(stmt().bind("callExpr"), &HandlerForCallExpr);
}

void ASTConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
    // 1. collect
    astContext = &Context;
    Matcher.matchAST(Context);
}

void Remove(std::string &str, const std::string &remove_str) {
    for (size_t i; (i = str.find(remove_str)) != std::string::npos;)
        str.replace(i, remove_str.length(), "");
}

std::string GetTypeName(clang::QualType type, clang::ASTContext *ctx) {
    type = type.getCanonicalType();
    auto baseName = type.getAsString(ctx->getLangOpts());
    Remove(baseName, "struct ");
    Remove(baseName, "class ");
    return baseName;
}

}// namespace luisa::clangcxx