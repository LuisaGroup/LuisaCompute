#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/runtime/device.h>
#include <luisa/vstl/ranges.h>
#include <luisa/ast/callable_library.h>
#include <filesystem>

namespace luisa::clangcxx {

struct LC_CLANGCXX_API Compiler {
    static compute::ShaderCreationInfo create_shader(
        const compute::ShaderOption &option,
        compute::Device &device,
        vstd::IRange<luisa::string_view>& defines,
        const std::filesystem::path &shader_path,
        const std::filesystem::path &include_path) LUISA_NOEXCEPT;
    static compute::CallableLibrary export_callables(
        compute::Device &device,
        vstd::IRange<luisa::string_view>& defines,
        const std::filesystem::path &shader_path,
        const std::filesystem::path &include_path) LUISA_NOEXCEPT;
    static void lsp_compile_commands(
        vstd::IRange<luisa::string_view>& defines,
        const std::filesystem::path &shader_dir,
        const std::filesystem::path &shader_relative_dir,
        const std::filesystem::path &include_path,
        luisa::vector<char> &result) LUISA_NOEXCEPT;
private:
    compute::ShaderOption option;
    static luisa::vector<luisa::string> compile_args(
                vstd::IRange<luisa::string_view>& defines,
        const std::filesystem::path &shader_path,
        const std::filesystem::path &include_path,
        bool is_lsp,
        bool is_export) LUISA_NOEXCEPT;
};

}// namespace luisa::clangcxx
