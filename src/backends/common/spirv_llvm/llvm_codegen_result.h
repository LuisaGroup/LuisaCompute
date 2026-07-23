#pragma once

#include <luisa/vstl/common.h>
#include <luisa/vstl/vector.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/function.h>
#include "../hlsl/shader_property.h"

namespace lc::llvm_codegen {

using namespace luisa;
using namespace luisa::compute;

/// Result of LLVM IR code generation for a single kernel.
/// Mirrors hlsl::CodegenResult and lc::spirv::SpirvResult.
struct LLVMCodegenResult {
    using Properties = vstd::vector<hlsl::Property>;

    luisa::vector<uint32_t> spv_bin;                                       // SPIR-V binary (words)
    Properties properties;                                                 // Binding properties (reuse hlsl::Property)
    vstd::vector<std::pair<vstd::string, Type const *>> printers;         // Print format info
    bool useTex2DBindless{false};                                          // Whether 2D bindless textures are used
    bool useTex3DBindless{false};                                          // Whether 3D bindless textures are used
    bool useBufferBindless{false};                                         // Whether bindless buffers are used
    LLVMCodegenResult() = default;
    LLVMCodegenResult(LLVMCodegenResult &&) = default;
    LLVMCodegenResult(LLVMCodegenResult const &) = delete;
};

} // namespace lc::llvm_codegen
