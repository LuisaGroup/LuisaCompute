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
    vstd::IRange<luisa::string_view>& defines,
    const std::filesystem::path &shader_path,
    const std::filesystem::path &include_path,
    bool is_lsp) {
    auto include_arg = "-I" + detail::path_to_string(include_path);
    auto const &output_path = context.runtime_directory();
    luisa::string output_arg = "--output=";
    output_arg += detail::path_to_string(output_path);

    luisa::string arg_list[] = {
        "-std=c++23",
        // swizzle uses reference member in union
        "-fms-extensions",
        "-Wno-microsoft-union-member-reference",
        std::move(include_arg)};
    luisa::vector<luisa::string> args_holder;
    size_t reserve_size = vstd::array_count(arg_list);
    if (!is_lsp) {
        luisa::string compile_arg_list[] = {
            "luisa_compiler",
            std::move(detail::path_to_string(shader_path)),
            std::move(output_arg),
            "--"};
        reserve_size += vstd::array_count(compile_arg_list);
        args_holder.reserve(reserve_size);
        vstd::push_back_func(args_holder, vstd::array_count(compile_arg_list), [&](size_t i) -> auto && {
            return std::move(compile_arg_list[i]);
        });
    } else {
        args_holder.reserve(reserve_size);
    }
    vstd::push_back_func(args_holder, vstd::array_count(arg_list), [&](size_t i) -> auto && {
        return std::move(arg_list[i]);
    });
    for(auto&& i : defines){
        auto d = luisa::string("-D");
        d += i;
        args_holder.emplace_back(std::move(d));
    }
    return args_holder;
}

compute::ShaderCreationInfo Compiler::create_shader(
    compute::Context const &context,
    luisa::compute::Device &device,
    vstd::IRange<luisa::string_view>& defines,
    const std::filesystem::path &shader_path,
    const std::filesystem::path &include_path) LUISA_NOEXCEPT {

    auto args_holder = compile_args(context, defines, shader_path, include_path, false);
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

void Compiler::lsp_compile_commands(
    compute::Context const &context,
    vstd::IRange<luisa::string_view>& defines,
    const std::filesystem::path &shader_dir,
    const std::filesystem::path &shader_relative_dir,
    const std::filesystem::path &include_path,
    luisa::vector<char> &result) {
    using namespace std::string_view_literals;
    auto args_holder = compile_args(context, defines, shader_relative_dir, include_path, true);
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