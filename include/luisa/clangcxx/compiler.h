#pragma once
#include "luisa/core/dll_export.h"
#include "luisa/runtime/device.h"

namespace luisa::clangcxx {

struct LC_CLANGCXX_API Compiler
{
    Compiler(const compute::ShaderOption &option);
    compute::ShaderCreationInfo create_shader(compute::Device& device) LUISA_NOEXCEPT;
    
    compute::ShaderOption option;
};

}// namespace lc::clangcxx
