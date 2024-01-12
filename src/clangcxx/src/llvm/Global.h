#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/runtime/device.h>
#include <luisa/clangcxx/compiler.h>
#include "Utils/OptionsParser.h"
#include "FrontendAction.h"

namespace luisa::clangcxx {

struct LC_CLANGCXX_API Global
{
    static llvm::cl::OptionCategory ToolCategoryOption;
    static llvm::cl::cat ToolCategory;
    static llvm::cl::extrahelp CommonHelp;
    static llvm::cl::opt<std::string> OutputDir;
};

}// namespace lc::clangcxx