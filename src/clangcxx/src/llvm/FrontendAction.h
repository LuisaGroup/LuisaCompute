#pragma once
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

namespace luisa::clangcxx {

class FrontendAction : public clang::ASTFrontendAction {
public:
    FrontendAction() = default;

    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) final;
};

}// namespace luisa::clangcxx