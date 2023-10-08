#include "clang/AST/Decl.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

#include "luisa/clangcxx/compiler.h"
#include "./consumer.hpp"

namespace luisa::clangcxx
{

namespace tooling = clang::tooling;
// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory ToolCategoryOption("meta options");
static llvm::cl::cat ToolCategory(ToolCategoryOption);

static llvm::cl::opt<std::string> Output(
    "output", llvm::cl::Optional,
    llvm::cl::desc("Specify database output directory, depending on extension"),
    ToolCategory, llvm::cl::value_desc("directory"));

class SSLFrontendAction : public clang::ASTFrontendAction
{
public:
    SSLFrontendAction() = default;

    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& CI, llvm::StringRef InFile) override
    {
        auto OutputPath = Output.hasArgStr() ? Output.getValue() : "./";
        auto& PP = CI.getPreprocessor();
        clang::SourceManager& SM = PP.getSourceManager();
        auto& LO = CI.getLangOpts();
        LO.CommentOpts.ParseAllComments = true;
        return std::make_unique<ASTConsumer>(std::move(OutputPath));
    }
};

}