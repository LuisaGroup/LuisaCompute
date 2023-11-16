#include "llvm/Global.h"
#include "llvm/FrontendAction.h"

namespace tooling = clang::tooling;

namespace luisa::clangcxx {

Compiler::Compiler(const compute::ShaderOption& option, compute::Function kernel)
    : option(option), kernel(kernel) 
{

}

template<typename T>
std::unique_ptr<FrontendActionFactory> newFrontendActionFactory2(luisa::compute::Device* device, compute::ShaderOption option, compute::Function kernel) {
    class SimpleFrontendActionFactory2 : public FrontendActionFactory {
    public:
        SimpleFrontendActionFactory2(luisa::compute::Device* device, compute::ShaderOption option, compute::Function kernel)
            : device(device), option(option), kernel(kernel)
        {

        }

        std::unique_ptr<clang::FrontendAction> create() override {
            return std::make_unique<T>(device, option, kernel);
        }

        luisa::compute::Device *device = nullptr;
        compute::ShaderOption option;
        compute::Function kernel;
    };
    return std::unique_ptr<FrontendActionFactory>(new SimpleFrontendActionFactory2(device, option, kernel));
}

compute::ShaderCreationInfo Compiler::create_shader(luisa::compute::Device &device) LUISA_NOEXCEPT {
    std::vector<std::string> args_holder = {
        "luisa_compiler",
        "./../../../LuisaCompute/src/clangcxx/shader/test_0.cpp",
        "--output=C:/LuisaCompute/src/clangcxx/shader",
        "--",
        "-std=c++20"};
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
    auto factory = newFrontendActionFactory2<luisa::clangcxx::FrontendAction>(&device, option, kernel);
    auto rc = Tool.run(factory.get());
    if (rc != 0) {
        // ...
    }
    return {};
}

}// namespace luisa::clangcxx