#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <filesystem>

namespace luisa::clangcxx {

struct LC_CLANGCXX_API Compiler {
    Compiler(const compute::ShaderOption &option);
    compute::ShaderCreationInfo create_shader(
        compute::Context const &context,
        compute::Device &device,
        const std::filesystem::path& shader_path) LUISA_NOEXCEPT;

    compute::ShaderOption option;
};

}// namespace luisa::clangcxx
