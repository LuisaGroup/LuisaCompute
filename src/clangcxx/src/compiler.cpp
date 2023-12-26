#include "llvm/Global.h"
#include "llvm/FrontendAction.h"
#include <luisa/vstl/vector.h>

namespace tooling = clang::tooling;

namespace luisa::clangcxx {
namespace detail {
string path_to_string(std::filesystem::path const &path) {
    auto str = luisa::to_string(path);
    for (auto &i : str) {
        if (i == '\\') {
            i = '/';
        }
    }
    return str;
}
}// namespace detail
Compiler::Compiler(const compute::ShaderOption &option)
    : option(option) {
}

template<typename T>
std::unique_ptr<FrontendActionFactory> newFrontendActionFactory2(luisa::compute::Device *device, compute::ShaderOption option) {
    class SimpleFrontendActionFactory2 : public FrontendActionFactory {
    public:
        SimpleFrontendActionFactory2(luisa::compute::Device *device, compute::ShaderOption &&option)
            : device(device), option(std::move(option)) {
        }

        std::unique_ptr<clang::FrontendAction> create() override {
            return std::make_unique<T>(device, option);
        }

        luisa::compute::Device *device = nullptr;
        compute::ShaderOption option;
    };
    return std::unique_ptr<FrontendActionFactory>(new SimpleFrontendActionFactory2(device, std::move(option)));
}

luisa::vector<luisa::string> Compiler::compile_args(
    compute::Context const &context,
    luisa::span<const luisa::string_view> defines,
    const std::filesystem::path &shader_path,
    const std::filesystem::path &include_path) {
    auto include_arg = "-I" + detail::path_to_string(include_path);
    auto const &output_path = context.runtime_directory();
    auto output_arg = detail::path_to_string(output_path);
    output_arg = "--output=" + output_arg;
    luisa::string arg_list[] = {
        "luisa_compiler",
        std::move(detail::path_to_string(shader_path)),
        std::move(output_arg),
        "--",
        "-std=c++20",
        // swizzle uses reference member in union
        "-fms-extensions",
        "-Wno-microsoft-union-member-reference",
        std::move(include_arg)};
    luisa::vector<luisa::string> args_holder;
    args_holder.reserve(vstd::array_count(arg_list) + defines.size());
    vstd::push_back_func(args_holder, vstd::array_count(arg_list), [&](size_t i) -> auto && {
        return std::move(arg_list[i]);
    });
    vstd::push_back_func(args_holder, defines.size(), [&](size_t i) {
        auto d = luisa::string("-D");
        d += defines[i];
        return d;
    });
    return args_holder;
}

compute::ShaderCreationInfo Compiler::create_shader(
    compute::Context const &context,
    luisa::compute::Device &device,
    luisa::span<const luisa::string_view> defines,
    const std::filesystem::path &shader_path,
    const std::filesystem::path &include_path) LUISA_NOEXCEPT {

    auto args_holder = compile_args(context, defines, shader_path, include_path);
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

luisa::string Compiler::lsp_compile_commands(
    compute::Context const &context,
    luisa::span<const luisa::string_view> defines,
    const std::filesystem::path &shader_dir,
    const std::filesystem::path &shader_relative_dir,
    const std::filesystem::path &include_path) {
    using namespace std::string_view_literals;
    auto args_holder = compile_args(context, defines, shader_relative_dir, include_path);
    luisa::string json;
    json += R"({"directory":")"sv;
    json += detail::path_to_string(shader_dir);
    json += R"(","arguments":[")"sv;
    json += detail::path_to_string(context.runtime_directory() / "clang.exe"sv);
    json += "\",";
    for (auto &i : args_holder) {
        json += "\"";
        json += i;
        json += "\",";
    }
    json.pop_back();
    json += R"(],"file":")"sv;
    json += detail::path_to_string(shader_relative_dir);
    json += "\"}";
    return json;
}

}// namespace luisa::clangcxx