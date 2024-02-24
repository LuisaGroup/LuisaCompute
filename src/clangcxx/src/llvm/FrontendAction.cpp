#include "Global.h"
#include "FrontendAction.h"
#include "ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"

namespace luisa::clangcxx {

std::unique_ptr<clang::ASTConsumer> FrontendAction::CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) {
    auto &LO = CI.getLangOpts();
    LO.CommentOpts.ParseAllComments = true;
    return std::make_unique<luisa::clangcxx::ASTConsumer>(device, option);
}
std::unique_ptr<clang::ASTConsumer> CallLibFrontendAction::CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) {
    auto &LO = CI.getLangOpts();
    LO.CommentOpts.ParseAllComments = true;
    return std::make_unique<luisa::clangcxx::ASTCallableConsumer>(lib);
}
}// namespace luisa::clangcxx