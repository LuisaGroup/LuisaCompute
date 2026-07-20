#pragma once

#include "llvm_codegen_result.h"

#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/resource.h>

namespace lc::llvm_codegen {

[[nodiscard]] LLVMCodegenResult compile_spirv(
    luisa::compute::Function kernel,
    const luisa::compute::ShaderOption &option);

}// namespace lc::llvm_codegen
