#pragma once
#include "luisa/core/dll_export.h"
#include "luisa/runtime/device.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

namespace luisa::clangcxx {

class FrontendAction : public clang::ASTFrontendAction {
public:
    FrontendAction(luisa::compute::Device* device, compute::ShaderOption option, compute::Function kernel)
        : clang::ASTFrontendAction(), device(device), option(option), kernel(kernel)
    {

    }

    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) final;
    
    luisa::compute::Device* device = nullptr;
    compute::ShaderOption option;
    compute::Function kernel;
};

}// namespace luisa::clangcxx