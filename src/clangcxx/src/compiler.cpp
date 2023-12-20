#include "llvm/Global.h"
#include "llvm/FrontendAction.h"

namespace tooling = clang::tooling;

namespace luisa::clangcxx {

Compiler::Compiler(const compute::ShaderOption &option)
    : option(option) {
}

template<typename T>
std::unique_ptr<FrontendActionFactory> newFrontendActionFactory2(luisa::compute::Device *device, compute::ShaderOption option) {
    class SimpleFrontendActionFactory2 : public FrontendActionFactory {
    public:
        SimpleFrontendActionFactory2(luisa::compute::Device *device, compute::ShaderOption option)
            : device(device), option(option) {
        }

        std::unique_ptr<clang::FrontendAction> create() override {
            return std::make_unique<T>(device, option);
        }

        luisa::compute::Device *device = nullptr;
        compute::ShaderOption option;
    };
    return std::unique_ptr<FrontendActionFactory>(new SimpleFrontendActionFactory2(device, option));
}

compute::ShaderCreationInfo Compiler::create_shader(
    compute::Context const &context,
    luisa::compute::Device &device,
    const std::filesystem::path& shader_path) LUISA_NOEXCEPT {
    auto output_path = context.runtime_directory() / "./../../src/clangcxx/shader";
    auto output_arg = output_path.string();
    output_arg = "--output=" + output_arg;
    luisa::vector<std::string> args_holder = {
        "luisa_compiler",
        std::move(shader_path.string()),
        std::move(output_arg),
        "--",
        "-std=c++20"};
    luisa::vector<const char *> args;
    args.reserve(args_holder.size());
    for (auto &arg : args_holder) {
        args.push_back(arg.c_str());
    }
    int argc = (int)args.size();
    auto ExpectedParser = OptionsParser::create(argc, args.data(), llvm::cl::ZeroOrMore, Global::ToolCategoryOption);
    if (!ExpectedParser) {
        // Fail gracefully for unsupported options.
        llvm::errs() << ExpectedParser.takeError();
        return compute::ShaderCreationInfo::make_invalid();
    }
    OptionsParser &OptionsParser = ExpectedParser.get();
    tooling::ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
    auto factory = newFrontendActionFactory2<luisa::clangcxx::FrontendAction>(&device, option);
    auto rc = Tool.run(factory.get());
    if (rc != 0) {
        // ...
    }
    return compute::ShaderCreationInfo::make_invalid();
}

}// namespace luisa::clangcxx