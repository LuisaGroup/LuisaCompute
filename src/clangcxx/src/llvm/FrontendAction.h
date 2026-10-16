#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/runtime/device.h>
#include <luisa/clangcxx/build_arguments.h>
#include <luisa/core/stl/memory.h>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

namespace luisa::clangcxx {

class FrontendAction : public clang::ASTFrontendAction {
public:
    FrontendAction(luisa::compute::Device *device, ShaderReflection *kernel_arg_reflect, compute::ShaderOption option)
        : clang::ASTFrontendAction(), device(device), kernel_arg_reflect(kernel_arg_reflect), option(option) {
    }

    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) final;

    luisa::compute::Device *device = nullptr;
    ShaderReflection *kernel_arg_reflect = nullptr;
    compute::ShaderOption option;
};
class CallLibFrontendAction : public clang::ASTFrontendAction {
public:
    CallLibFrontendAction(compute::CallableLibrary *lib)
        : clang::ASTFrontendAction(), lib(lib) {
    }
    compute::CallableLibrary *lib;
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef InFile) final;
};

}// namespace luisa::clangcxx