#include "llvm/Global.h"
#include "llvm/FrontendAction.h"
#include <luisa/vstl/vector.h>
#include <luisa/vstl/ranges.h>

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

template<typename T>
std::unique_ptr<FrontendActionFactory> newFrontendActionFactory3(compute::CallableLibrary *lib) {
    class SimpleFrontendActionFactory3 : public FrontendActionFactory {
    public:
        SimpleFrontendActionFactory3(
            compute::CallableLibrary *lib) : lib(lib) {
        }

        std::unique_ptr<clang::FrontendAction> create() override {
            return std::make_unique<T>(lib);
        }
        compute::CallableLibrary *lib;
    };
    return std::unique_ptr<FrontendActionFactory>(new SimpleFrontendActionFactory3(lib));
}

luisa::vector<luisa::string> Compiler::compile_args(
    vstd::IRange<luisa::string_view> &defines,
    const std::filesystem::path &shader_path,
    vstd::IRange<luisa::string> &include_paths,
    bool is_lsp,
    bool is_export) LUISA_NOEXCEPT {
    luisa::vector<luisa::string> arg_list = {
        "-std=c++23",
        // swizzle uses reference member in union
        "-fms-extensions",
        "-Wno-microsoft-union-member-reference"};
    for (auto i : include_paths) {
        for (auto &c : i) {
            if (c == '\\') c = '/';
        }
        arg_list.emplace_back(luisa::string{"-I"} + std::move(i));
    }
    luisa::vector<luisa::string> args_holder;
    if (!is_lsp) {
        luisa::string compile_arg_list[] = {
            "luisa_compiler",
            std::move(detail::path_to_string(shader_path)),
            "--"};
        vstd::push_back_func(args_holder, vstd::array_count(compile_arg_list), [&](size_t i) -> auto && {
            return std::move(compile_arg_list[i]);
        });
    }
    vstd::push_back_func(args_holder, arg_list.size(), [&](size_t i) -> auto && {
        return std::move(arg_list[i]);
    });
    for (auto &&i : defines) {
        auto d = luisa::string("-D");
        d += i;
        args_holder.emplace_back(std::move(d));
    }
    if (is_export) {
        args_holder.emplace_back("-D_EXPORT");
    }
    return args_holder;
}

bool Compiler::create_shader(
    const compute::ShaderOption &option,
    luisa::compute::Device &device,
    vstd::IRange<luisa::string_view> &defines,
    const std::filesystem::path &shader_path,
    vstd::IRange<luisa::string> &include_paths) LUISA_NOEXCEPT {

    auto args_holder = compile_args(defines, shader_path, include_paths, false, false);
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
        return false;
    }
    OptionsParser &OptionsParser = ExpectedParser.get();
    tooling::ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
    auto factory = newFrontendActionFactory2<luisa::clangcxx::FrontendAction>(&device, option);
    // Callable export
    // auto factory = newFrontendActionFactory3<luisa::clangcxx::CallLibFrontendAction>(option.name);
    auto rc = Tool.run(factory.get());
    if (rc != 0) {
        return false;
    }
    return true;
}
compute::CallableLibrary Compiler::export_callables(
    compute::Device &device,
    vstd::IRange<luisa::string_view> &defines,
    const std::filesystem::path &shader_path,
    vstd::IRange<luisa::string> &include_paths) LUISA_NOEXCEPT {
    compute::CallableLibrary lib;
    auto args_holder = compile_args(defines, shader_path, include_paths, false, true);
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
        return lib;
    }
    OptionsParser &OptionsParser = ExpectedParser.get();
    tooling::ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
    auto factory = newFrontendActionFactory3<luisa::clangcxx::CallLibFrontendAction>(&lib);
    auto rc = Tool.run(factory.get());
    if (rc != 0) {
        // ...
    }
    return lib;
}

void Compiler::lsp_compile_commands(
    vstd::IRange<luisa::string_view> &defines,
    const std::filesystem::path &shader_dir,
    const std::filesystem::path &shader_relative_dir,
    vstd::IRange<luisa::string> &include_paths,
    luisa::vector<char> &result) LUISA_NOEXCEPT {
    using namespace std::string_view_literals;
    auto args_holder = compile_args(defines, shader_relative_dir, include_paths, true, false);
    auto add = [&]<typename T>(T c) {
        if constexpr (std::is_same_v<T, char const *>) {
            vstd::push_back_all(result, span<char const>(c, strlen(c)));
        } else if constexpr (std::is_same_v<T, char>) {
            result.emplace_back(c);
        } else {
            vstd::push_back_all(result, span<char const>(c.data(), c.size()));
        }
    };
    add(R"({"directory":")"sv);
    add(detail::path_to_string(shader_dir));
    add(R"(","arguments":[")"sv);
    add("clang.exe"sv);
    add("\","sv);
    for (auto &i : args_holder) {
        add('"');
        add(i);
        add("\","sv);
    }
    result.pop_back();
    add(R"(],"file":")"sv);
    add(detail::path_to_string(shader_relative_dir));
    add("\"}");
}

}// namespace luisa::clangcxx