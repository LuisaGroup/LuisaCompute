#pragma once
#include "luisa/core/dll_export.h"
#include "luisa/runtime/device.h"

namespace luisa::clangcxx {

struct LC_CLANGCXX_API Compiler
{
    void compile(luisa::compute::Device& device) LUISA_NOEXCEPT;
};

}// namespace lc::clangcxx
