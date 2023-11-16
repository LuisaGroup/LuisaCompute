#include "Global.h"
#include "FrontendAction.h"
#include "ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"

namespace luisa::clangcxx {

std::unique_ptr<clang::ASTConsumer> FrontendAction::CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile)
{
    auto& Output = Global::OutputDir;
    auto OutputPath = Output.hasArgStr() ? Output.getValue() : "./";
    auto &PP = CI.getPreprocessor();
    clang::SourceManager &SM = PP.getSourceManager();
    auto &LO = CI.getLangOpts();
    LO.CommentOpts.ParseAllComments = true;
    return std::make_unique<luisa::clangcxx::ASTConsumer>(OutputPath);
}

}