#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>

namespace luisa::clangcxx {

struct LC_CLANGCXX_API Compiler {
    Compiler(const compute::ShaderOption &option);
    compute::ShaderCreationInfo create_shader(
        compute::Context const &context,
        compute::Device &device) LUISA_NOEXCEPT;

    compute::ShaderOption option;
};

}// namespace luisa::clangcxx
