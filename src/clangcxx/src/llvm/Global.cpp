#include "Global.h"

namespace luisa::clangcxx {

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
llvm::cl::OptionCategory Global::ToolCategoryOption("meta options");
llvm::cl::cat Global::ToolCategory(ToolCategoryOption);

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
llvm::cl::extrahelp Global::CommonHelp = llvm::cl::extrahelp(tooling::CommonOptionsParser::HelpMessage);

llvm::cl::opt<std::string> Global::OutputDir(
    "output", llvm::cl::Optional,
    llvm::cl::desc("Specify shader output directory, depending on extension"),
    ToolCategory, llvm::cl::value_desc("directory"));

}// namespace luisa::clangcxx