#include "llvm/Global.h"
#include "llvm/FrontendAction.h"

namespace tooling = clang::tooling;

namespace luisa::clangcxx {

Compiler::Compiler(const compute::ShaderOption &option, compute::Function kernel) 
{
}

compute::ShaderCreationInfo Compiler::create_shader(luisa::compute::Device &device) LUISA_NOEXCEPT 
{
    std::vector<std::string> args_holder = {
        "luisa_compiler",
        "C:/GitHub/LuisaCompute/src/clangcxx/shader/test_0.cpp",
        "--output=C:/LuisaCompute/src/clangcxx/shader",
        "--",
        "-std=c++20"
    };
    std::vector<const char *> args;
    args.reserve(args_holder.size());
    for (auto &arg : args_holder) {
        args.push_back(arg.c_str());
    }
    int argc = (int)args.size();
    auto ExpectedParser = OptionsParser::create(argc, args.data(), llvm::cl::ZeroOrMore, Global::ToolCategoryOption);
    if (!ExpectedParser) {
        // Fail gracefully for unsupported options.
        llvm::errs() << ExpectedParser.takeError();
        return {};
    }
    OptionsParser &OptionsParser = ExpectedParser.get();
    tooling::ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
    auto rc = Tool.run(tooling::newFrontendActionFactory<FrontendAction>().get());
    if (rc != 0)
    {
        // ...
    }
    return {};
}

}// namespace luisa::clangcxx